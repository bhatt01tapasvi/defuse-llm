import torch
import os
import json
import re
import numpy as np
from src.util import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRUNE_DIR = os.path.join(RESULTS_DIR, "pruneNeurons")
#MAX_FORWARD_PROXY = os.path.join(PRUNING_DIR, "maxProxy")

class NeuronDefuser:
    def __init__(self, maskingStep: int=0, per_layer_topk: dict=None, ema_decay: float=None, ranking_method: str='combined', prune_strategy: str='topk', device: str='cuda'):
        """
        Args:
            maskingStep: Step at which to start applying masks
            per_layer_topk: Dict mapping layer_name -> topK value. 
                           Use -1 to skip pruning for that layer.
            ema_decay: Decay factor for EMA (0.0 to 1.0)
            ranking_method: One of ['max', 'mean', 'combined', 'product']
            device: Device to run on
        """
        self.currIteration = 0
        self.maskingStep = maskingStep
        self.device = device
        self.ema_decay = ema_decay  # Decay factor for EMA
        self.ranking_method = ranking_method
        self.prune_strategy = prune_strategy  # 'topk' or 'threshold'

        # Store the original per_layer_topk config
        self.per_layer_topk_config = per_layer_topk if per_layer_topk is not None else {}

        self.per_layer_topk = {}
        self.layer_name_to_number = {}  # Map layer names to their numbers
        self.layer_number_to_name = {}  # Map layer numbers to their names
        self.pre_mlp_cache = {}   # Temporary storage per layer
        self.mlp_impact_history = {}  # Accumulated metrics per layer
        #   Format: {layer_name: {'cosine': [values], 'delta_ratio': [values]}}

        self.forward_proxies_max = {}  # Store max forward proxies per layer
        self.forward_proxies_mean = {}  # Store mean forward proxies per layer
        self.ema_max = {}  # Store EMA per layer
        self.ema_mean = {}  # Store EMA per layer
        self.ema_activations = {}  # Store EMA of activations per layer
        self.masks = {}  # Store per-layer masks

        # Storage for neuron statistics (populated during defuse_neurons)
        self.neuron_stats = {}  # layer_name -> {strong_ids, weakly_strong_ids, weak_count, topk_max, topk_mean}

    def _extract_layer_number(self, layer_name: str) -> int:
        """
        Extract layer number from layer name.
        Handles format: 'layer_0', 'layer_1', 'layer_2', etc.
        """
        
        # Match 'layer_' followed by digits
        match = re.search(r'layer_(\d+)', layer_name)
        if match:
            layer_num = int(match.group(1))
            return layer_num
        
        return -1

    def _register_layer(self, layer_name: str):
        """Register a layer and map it to the pruning configuration."""
        if layer_name in self.layer_name_to_number:
            return  # Already registered
        
        layer_num = self._extract_layer_number(layer_name)
        self.layer_name_to_number[layer_name] = layer_num
        self.layer_number_to_name[layer_num] = layer_name
        
        # Map the topk value based on configuration
        topk_value = -1  # Default: no pruning
        
        # Check if configuration uses layer numbers (int keys) or layer names (str keys)
        if layer_num in self.per_layer_topk_config:
            topk_value = self.per_layer_topk_config[layer_num]
        elif layer_name in self.per_layer_topk_config:
            topk_value = self.per_layer_topk_config[layer_name]
        else:
            print(f"DEBUG: No match found for layer {layer_num} or {layer_name}")
        
        self.per_layer_topk[layer_name] = topk_value
        print(f"DEBUG: Set topk for {layer_name} to {topk_value}")

    def populate_forward_proxy(self, layer_name: str, weight: torch.Tensor, embedding_weights: torch.Tensor):
        self._register_layer(layer_name)

        # Ensure tensors are on correct device
        weight = weight.to(self.device)
        embedding_weights = embedding_weights.to(self.device)

        # Calculate forward proxy and keep it on GPU
        forward_proxy = (weight @ embedding_weights.T).detach()  # Remove .cpu().numpy()
        self.forward_proxies_max[layer_name] = torch.max(torch.abs(forward_proxy), dim=1)[0]
        self.forward_proxies_mean[layer_name] = torch.mean(torch.abs(forward_proxy), dim=1)
        
        del forward_proxy
        torch.cuda.empty_cache()

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _compute_combined_score(self, act_score: torch.Tensor, max_scores: torch.Tensor, 
                                mean_scores: torch.Tensor) -> torch.Tensor:
        if self.ranking_method == 'magnitude':
            # Use absolute mean scores
            return act_score
        elif self.ranking_method == 'max':
            return max_scores
        elif self.ranking_method == 'mean':
            return mean_scores
        elif self.ranking_method == 'combined':
            # Normalize both scores and take weighted sum
            #norm_max = self._normalize_scores(max_scores)
            #norm_mean = self._normalize_scores(mean_scores)
            #return norm_max + norm_mean  # Equal weighting
            return max_scores + mean_scores
        elif self.ranking_method == 'product':
            # Multiply normalized scores (favors neurons high in both)
            #norm_max = self._normalize_scores(max_scores)
            #norm_mean = self._normalize_scores(mean_scores)
            #return norm_max * norm_mean
            return max_scores * mean_scores
        else:
            raise ValueError(f"Unknown ranking method: {self.ranking_method}")

    def defuse_neurons(self, layer_name: str, activations3: torch.Tensor):
        """
        Deactivates neurons in the activations tensor that have mean activation below the threshold.
        
        TODO: There are two appraoches here:
        We process each new token in each forward pass. 
        Or we process all the tokens in one go just before we have to start masking and then judge what we need and what we don't.
        For now, I am implementing the first approach.
        """

        if self.maskingStep is None:
            return activations3  # No masking step defined, return unchanged
        
        if self.currIteration > self.maskingStep:
            if layer_name in self.masks and self.masks[layer_name] is not None:
                return activations3 * self.masks[layer_name]
            return activations3

        batch_size, seq_len, hidden_dim = activations3.shape
        activations = activations3[0]

        if self.currIteration == 0:
            # Multiply activations with proxy values (broadcasting)
            last_gen_forward_proxy_max = torch.abs(activations) * self.forward_proxies_max[layer_name]
            last_gen_forward_proxy_mean = torch.abs(activations) * self.forward_proxies_mean[layer_name]
            
            seq_length = last_gen_forward_proxy_max.shape[0]

            # Check if ema_decay is None (use L2 norm aggregation)
            if self.ema_decay is None:
                # L2 norm: sqrt(sum of squares) for each neuron
                self.ema_activations[layer_name] = torch.sqrt(
                    torch.sum(torch.abs(activations) ** 2, dim=0)
                )
                self.ema_max[layer_name] = torch.sqrt(
                    torch.sum(last_gen_forward_proxy_max ** 2, dim=0)
                )
                self.ema_mean[layer_name] = torch.sqrt(
                    torch.sum(last_gen_forward_proxy_mean ** 2, dim=0)
                )
            else:
                # Apply exponential weighting
                weights = self.ema_decay ** torch.arange(seq_length - 1, -1, -1, 
                                                      dtype=torch.float32, 
                                                      device=self.device)

                weights = weights / weights.sum()
                self.ema_activations[layer_name] = torch.sum(
                    torch.abs(activations) * weights.unsqueeze(1), 
                    dim=0
                )
                self.ema_max[layer_name] = torch.sum(
                    last_gen_forward_proxy_max * weights.unsqueeze(1), 
                    dim=0
                )
                self.ema_mean[layer_name] = torch.sum(
                    last_gen_forward_proxy_mean * weights.unsqueeze(1), 
                    dim=0
                )

        elif self.currIteration <= self.maskingStep: # When currIter == maskingStep, we make the final update and mask, then don't compute ema again.
            # Multiply activations with proxy values (broadcasting)
            last_gen_forward_proxy_max = torch.abs(activations[-1]) * self.forward_proxies_max[layer_name]  # Shape: (num_tokens, embed_dim)
            last_gen_forward_proxy_mean = torch.abs(activations[-1]) * self.forward_proxies_mean[layer_name]
            
            # Check if ema_decay is None (use L2 norm accumulation)
            if self.ema_decay is None:
                # L2 norm accumulation: sqrt(sum_old^2 + new_value^2)
                # This is equivalent to: new_l2 = sqrt(old_l2^2 + new^2)
                self.ema_max[layer_name] = torch.sqrt(
                    self.ema_max[layer_name] ** 2 + last_gen_forward_proxy_max ** 2
                )
                self.ema_mean[layer_name] = torch.sqrt(
                    self.ema_mean[layer_name] ** 2 + last_gen_forward_proxy_mean ** 2
                )
            else:
                # Update EMA: ema_new = decay * ema_old + (1 - decay) * new_value
                print(f"FIX ME PLS")
                self.ema_max[layer_name] = self.ema_decay * self.ema_max[layer_name] + last_gen_forward_proxy_max
                self.ema_mean[layer_name] = self.ema_decay * self.ema_mean[layer_name] + last_gen_forward_proxy_mean

        is_last_layer = (layer_name == list(self.forward_proxies_max.keys())[-1])
        
        # Keep a count of the iteration we are at.
        if self.currIteration < self.maskingStep:
            if is_last_layer:
                self.currIteration += 1
            return activations3
        elif self.currIteration == self.maskingStep:
            layer_topk = self.per_layer_topk.get(layer_name, -1)  # Get configured value
            layer_num = self.layer_name_to_number.get(layer_name, -1)
    
            # Check if this layer should be pruned
            if layer_topk == -2:  # AUTO_TOPK_SENTINEL
                layer_topk = self.compute_adaptive_topk(layer_name=layer_name, hidden_dim=hidden_dim, total_prune_percent=50.0)
                print(f"Auto-calculated topk for layer {layer_num}: {layer_topk}")
            else:
                print(f"Using manual topk for layer {layer_num}: {layer_topk}")
            
            if layer_topk == -1:
                # Don't prune this layer - keep all neurons
                print(f"Skipping pruning for layer {layer_num}: {layer_name}")
                self.masks[layer_name] = None  # No mask means keep all
            else:
                # Prune this layer
                combined_score = self._compute_combined_score(
                    act_score=self.ema_activations[layer_name],
                    max_scores=self.ema_max[layer_name], 
                    mean_scores=self.ema_mean[layer_name]
                )
                
                # Get top-K neurons based on combined ranking
                if self.prune_strategy == 'topk':
                    topk_values, topk_indices = torch.topk(combined_score, layer_topk)
                elif self.prune_strategy == 'auto':
                    threshold = combined_score.mean()
                    min_val = combined_score.min()
                    max_val = combined_score.max()
                    topk_indices = (combined_score >= threshold).nonzero(as_tuple=True)[0]
                    topk_values = combined_score[topk_indices]
                    sorted_order = torch.argsort(topk_values, descending=True)
                    topk_indices = topk_indices[sorted_order]
                    topk_values = topk_values[sorted_order]
                
                # Create mask: keep only top-K neurons
                mask = torch.zeros(hidden_dim, dtype=activations3.dtype, device=self.device)
                mask[topk_indices] = 1.0
                self.masks[layer_name] = mask
                
                # Store detailed statistics
                actual_kept = len(topk_indices) if self.prune_strategy == 'auto' else layer_topk

                self.neuron_stats[layer_name] = {
                    'layer_number': layer_num,
                    'total_neurons': hidden_dim,
                    'neurons_kept': actual_kept,  # Use actual count
                    'neurons_pruned': hidden_dim - actual_kept,
                    'pruning_percentage': ((hidden_dim - actual_kept) / hidden_dim) * 100,
                    'kept_neuron_ids': topk_indices.cpu().tolist(),
                    'kept_neuron_scores': topk_values.cpu().tolist(),
                    'prune_strategy': self.prune_strategy,
                    'threshold': threshold.item() if self.prune_strategy == 'auto' else None,
                    'min_value': min_val.item() if self.prune_strategy == 'auto' else None,
                    'max_value': max_val.item() if self.prune_strategy == 'auto' else None
                }
                
        # Increment iteration counter
        if is_last_layer:
            self.currIteration += 1

        # Apply mask to activations (if mask exists for this layer)
        if layer_name in self.masks and self.masks[layer_name] is not None:
            activations3 *= self.masks[layer_name]
            #print the dimensions of activations3
            print(f"Applied mask to layer {layer_name}, activations shape: {activations3.shape}")

            # Check the mast is applied properly by counting the number of 0 and that they match the masked neurons.
            num_masked = (self.masks[layer_name] == 0).sum().item()
            num_actually_masked = (activations3[-1][:, self.masks[layer_name] == 0] == 0).all(dim=0).sum().item()
            assert num_masked == num_actually_masked, f"Masking error in layer {layer_name}: expected {num_masked} masked neurons, but only {num_actually_masked} neurons are actually zero."
        
        return activations3

    def cache_pre_mlp(self, layer_name, activation):
        """Called by post_attention_layernorm hook"""
        if self.currIteration > self.maskingStep:
            return
        self.pre_mlp_cache[layer_name] = activation

    def calculate_mlp_impact(self, layer_name, post_mlp_activation):
        """Called by mlp_down post-hook"""
        if self.currIteration > self.maskingStep:
            return
        if layer_name not in self.pre_mlp_cache:
            return

        # Store metrics
        if layer_name not in self.mlp_impact_history:
            self.mlp_impact_history[layer_name] = {'cosine': [], 'delta': []}

        pre_mlp = self.pre_mlp_cache[layer_name]
        post_mlp = post_mlp_activation
        
        cosine = cosine_similarity(pre_mlp, post_mlp)
        pre = np.asarray(pre_mlp, dtype=np.float64)
        post = np.asarray(post_mlp, dtype=np.float64)
        if pre.ndim != post.ndim:
            raise ValueError(f"relative_mlp_impact expects same ndim (got {pre.ndim} and {post.ndim})")

        delta = post - pre

        if pre.ndim == 1:
            norm_pre = np.linalg.norm(pre)
            norm_delta = np.linalg.norm(delta)
            relative_impact = 0.0 if norm_pre == 0.0 else float(norm_delta / norm_pre)
            self.mlp_impact_history[layer_name]['delta'].append(relative_impact)
        elif pre.ndim == 2:
            # Per-token relative impact along last axis
            norm_pre = np.linalg.norm(pre, axis=-1)
            norm_delta = np.linalg.norm(delta, axis=-1)
            rel = np.zeros_like(norm_pre)
            valid = norm_pre > 0.0
            rel[valid] = norm_delta[valid] / norm_pre[valid]
            # Replace any non-finite with 0.0 for safety
            rel = np.where(np.isfinite(rel), rel, 0.0)
            self.mlp_impact_history[layer_name]['delta'].extend(rel.tolist())
        else:
            raise ValueError("relative_mlp_impact supports only 1D or 2D inputs")

        # If scalar (1D inputs), append single value; if array (2D inputs), extend per token
        if np.isscalar(cosine):
            self.mlp_impact_history[layer_name]['cosine'].append(float(cosine))
        else:
            sims_arr = np.asarray(cosine, dtype=np.float64).ravel()
            self.mlp_impact_history[layer_name]['cosine'].extend(sims_arr.tolist())
        
        # Clear cache
        del self.pre_mlp_cache[layer_name]

    def compute_adaptive_topk(self, layer_name, hidden_dim: int = 4096, total_prune_percent: float = 50.0,
                we: float = 0.3, wl: float = 0.15, alpha: float = 0.25, beta: float = 0.35,
                p_min: float = 0.10, p_max: float = 0.90, epsilon: float = 1e-6,
                max_iterations: int = 10):
        """
        Rank layers and calculate pruning ratios using depth-aware protection.
        
        Implements the 9-step pruning strategy:
        1. Calculate per-layer functional importance S_ℓ
        2. Normalize importance to [0,1] → Ŝ_ℓ
        3. Calculate relative pruning pressure R_ℓ = 1 - Ŝ_ℓ
        4. Apply depth-aware protection factor D_ℓ
        5. Compute final pruning pressure P_ℓ = R_ℓ · D_ℓ
        6. Allocate pruning budget proportionally → p̃_ℓ
        7. Enforce safety bounds [p_min, p_max] → p_ℓ^(0)
        8. Iteratively redistribute residual budget Δ
        9. Return final pruning ratios satisfying constraints
        
        Args:
            model_name: Name of the model
            dataset_name: Name of the dataset
            total_prune_percent: Target total pruning percentage ρ (e.g., 50.0 for 50%)
            we: Early protection width (default: 0.2)
            wl: Late protection width (default: 0.15)
            alpha: Early minimum scaling factor (default: 0.25)
            beta: Late minimum scaling factor (default: 0.35)
            p_min: Minimum pruning ratio per layer (default: 0.10)
            p_max: Maximum pruning ratio per layer (default: 0.90)
            epsilon: Small constant for numerical stability (default: 1e-6)
            max_iterations: Maximum iterations for budget correction (default: 100)
        """
        """Calculate topk based on accumulated impact metrics"""
        pass
        # if self.per_layer_topk[layer_name] == -2:
        #     num_layers = len(self.mlp_impact_history)

        #     if num_layers == 0:
        #         print("WARNING: No MLP impact history available for auto-topk calculation!")
        #         return

        #     avg_cosine = []
        #     avg_ratio_norm = []
        #     for layer_name, metrics in self.mlp_impact_history:
        #         avg_cosine.append(np.mean(metrics['cosine']))
        #         avg_ratio_norm.append(np.mean(metrics['delta']))
        #     layers = np.arange(num_layers)
        #     avg_cosine = np.array(avg_cosine)
        #     avg_ratio_norm = np.array(avg_ratio_norm)
            
        #     L = num_layers
        #     rho = total_prune_percent / 100.0
        #     B = rho * L  # Total pruning budget
            
        #     print(f"\n  {'='*80}")
        #     print(f"  Depth-Aware Pruning Strategy")
        #     print(f"  {'='*80}")
        #     print(f"  Target: {total_prune_percent}% total pruning (B = ρL = {B:.2f})")
        #     print(f"  Layers: L = {L}")
        #     print(f"  Config: we={we}, wl={wl}, α={alpha}, β={beta}, p_min={p_min}, p_max={p_max}")
            
        #     # ==================== STEP 1: Per-layer functional importance ====================
        #     # S_ℓ = (1 - cos_ℓ) · ||Δx_ℓ|| / ||x_ℓ,pre||
        #     S = (1 - avg_cosine) * (avg_ratio_norm)
            
        #     # ==================== STEP 2: Normalize importance ====================
        #     # Ŝ_ℓ = (S_ℓ - min S_j) / (max S_j - min S_j + ε)
        #     S_min = np.min(S)
        #     S_max = np.max(S)
        #     S_hat = (S - S_min) / (S_max - S_min + epsilon)
            
        #     # ==================== STEP 3: Relative pruning pressure ====================
        #     # R_ℓ = 1 - Ŝ_ℓ
        #     R = 1 - S_hat
            
        #     # ==================== STEP 4: Depth-aware protection factor ====================
        #     # z_ℓ = ℓ / (L - 1)
        #     z = layers / (L - 1)
        #     D = np.ones(num_layers)
            
        #     for i in range(num_layers):
        #         if z[i] < we:
        #             # Early layers: α + (1 - α) * z_ℓ / w_e
        #             D[i] = alpha + (1 - alpha) * (z[i] / we)
        #         elif z[i] > (1 - wl):
        #             # Late layers: β + (1 - β) * (1 - z_ℓ) / w_l
        #             D[i] = beta + (1 - beta) * ((1 - z[i]) / wl)
        #         else:
        #             # Middle layers: no protection
        #             D[i] = 1.0
            
        #     # ==================== STEP 5: Final pruning pressure ====================
        #     # P_ℓ = R_ℓ · D_ℓ
        #     P_pressure = R * D
            
        #     # ==================== STEP 6: Budgeted pruning allocation ====================
        #     # p̃_ℓ = B · P_ℓ / Σ_j P_j
        #     P_sum = np.sum(P_pressure)
        #     p_tilde = B * (P_pressure / P_sum)
            
        #     # ==================== STEP 7: Enforce safety bounds ====================
        #     # p_ℓ^(0) = clip(p̃_ℓ, p_min, p_max)
        #     p_0 = np.clip(p_tilde, p_min, p_max)
            
        #     initial_sum = np.sum(p_0)
        #     initial_delta = B - initial_sum
            
        #     # ==================== STEP 8: Budget correction ====================
        #     # Iteratively redistribute Δ among layers not at p_max (when Δ > 0)
        #     p = p_0.copy()
        #     iteration = 0
            
        #     while iteration < max_iterations:
        #         current_sum = np.sum(p)
        #         delta = B - current_sum
                
        #         # Check convergence
        #         if abs(delta) < epsilon:
        #             print(f"    ✓ Converged at iteration {iteration}: |Δ| = {abs(delta):.9f} < ε")
        #             break
                
        #         # Determine which layers can be adjusted based on sign of delta
        #         if delta > 0:
        #             # Need to prune MORE: only adjust layers below p_max (can increase pruning)
        #             available_mask = p < (p_max - epsilon)
        #             operation = "increase pruning"
        #         else:
        #             # Need to prune LESS: only adjust layers above p_min (can decrease pruning)
        #             available_mask = p > (p_min + epsilon)
        #             operation = "decrease pruning"
                
        #         if not np.any(available_mask):
        #             if delta > 0:
        #                 print(f"    ⚠ All layers at p_max. Cannot prune more.")
        #             else:
        #                 print(f"    ⚠ All layers at p_min. Cannot prune less.")
        #             print(f"      Final sum: {current_sum:.6f}, Target: {B:.6f}, Gap: {delta:.6f}")
        #             break
                
        #         # Weighted redistribution: p_ℓ = p_ℓ + Δ · [indicator · P_ℓ] / Σ[indicator · P_j]
        #         available_P = P_pressure * available_mask
        #         P_available_sum = np.sum(available_P)
                
        #         if P_available_sum > epsilon:
        #             redistribution = delta * (available_P / P_available_sum)
        #             p = p + redistribution
                    
        #             # Re-clip to bounds
        #             p = np.clip(p, p_min, p_max)
        #         else:
        #             # Fallback: uniform redistribution if no pressure weights available
        #             num_available = np.sum(available_mask)
        #             redistribution = delta / num_available
        #             p[available_mask] += redistribution
        #             p = np.clip(p, p_min, p_max)
                
        #         iteration += 1
                
        #         if iteration % 10 == 0:
        #             print(f"    Iteration {iteration}: Δ = {delta:.9f}, {operation}, available = {np.sum(available_mask)}")
            
        #     if iteration >= max_iterations:
        #         print(f"    ⚠ Reached max iterations ({max_iterations})")
            
        #     # ==================== STEP 9: Final results ====================
        #     final_sum = np.sum(p)
        #     final_percent = (final_sum / L) * 100
            
        #     print(f"\n  Step 9: Final pruning ratios")
        #     print(f"    Sum: {final_sum:.6f} / {B:.6f} (satisfaction: {(final_sum/B)*100:.2f}%)")
        #     print(f"    Average per-layer: {final_percent:.2f}%")
            
        #     # Convert to percentages for display
        #     p_percent = p * 100
        #     keep_percent = 100 - p_percent
            
        #     # ==================== Detailed Output ====================
        #     print(f"\n  {'='*115}")
        #     print(f"  Layer-wise Breakdown")
        #     print(f"  {'='*115}")
        #     print(f"  {'Layer':<6} {'z_ℓ':<8} {'S_ℓ':<10} {'Ŝ_ℓ':<10} {'R_ℓ':<10} {'D_ℓ':<10} {'P_ℓ':<10} {'p̃_ℓ':<10} {'p_ℓ':<10} {'Prune%':<10} {'Keep%':<10}")
        #     print(f"  {'-'*115}")
            
        #     for i in range(num_layers):
        #         print(f"  {layers[i]:<6} {z[i]:<8.4f} {S[i]:<10.6f} {S_hat[i]:<10.6f} {R[i]:<10.6f} {D[i]:<10.6f} "
        #             f"{P_pressure[i]:<10.6f} {p_tilde[i]:<10.6f} {p[i]:<10.6f} {p_percent[i]:<10.2f} {keep_percent[i]:<10.2f}")
            
        #     # ==================== Summary Statistics ====================
        #     print(f"\n  {'='*80}")
        #     print(f"  Summary Statistics")
        #     print(f"  {'='*80}")
        #     print(f"  Target Total Pruning:     {total_prune_percent:.2f}%")
        #     print(f"  Actual Total Pruning:     {final_percent:.2f}%")
            
        #     #TODO: Check this logic tomorrow!
        #     self.per_layer_topk[layer_name] = int(round((1 - p[self.layer_name_to_number[layer_name]]) * hidden_dim))
        # else:
        #     print(f"SAFETY HARNESS(will not run ideally): Layer {layer_name} has already been updated with the auto pruning calculations.")
        #     return self.per_layer_topk[layer_name]

    def save_results(self, filename_prefix="neuron_masking", output_dir=None):
        """Save the stored neuron statistics to files."""
        if output_dir is None:
            output_dir = PRUNE_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        summary_data = {
            'iteration': self.currIteration - 1,
            'masking_step': self.maskingStep,
            'ema_decay': self.ema_decay,
            'ranking_method': self.ranking_method,
            'per_layer_topk_config': self.per_layer_topk_config,
            'layers': {}
        }

        topk_neuron_data = {}

        for layer_name, stats in self.neuron_stats.items():
            # Create layer summary without full score arrays
            summary_data['layers'][layer_name] = {
                'layer_number': stats['layer_number'],
                'total_neurons': stats['total_neurons'],
                'neurons_kept': stats['neurons_kept'],
                'neurons_pruned': stats['neurons_pruned'],
                'pruning_percentage': stats['pruning_percentage'],
            }

            topk_neuron_data[layer_name] = {
                'kept_neuron_ids': stats['kept_neuron_ids'],
                'kept_neuron_scores': stats['kept_neuron_scores']
            }
        
        # Save summary JSON
        json_file = os.path.join(output_dir, f"{filename_prefix}_summary.json")
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Save topk neuron data separately
        topk_json_file = os.path.join(output_dir, f"{filename_prefix}_topk_neurons.json")
        with open(topk_json_file, 'w') as f:
            json.dump(topk_neuron_data, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Neuron ranking results saved:")
        print(f"  Summary: {json_file}")
        print(f"  Top-K Neurons: {topk_json_file}")
        print(f"{'='*80}\n")


## COMMENTED PORTION FOR L1
#
        #     if self.ema_decay is None:
        #         # L1 normalize per token (across neurons), then L2 aggregate across tokens
                
        #         # L1 norm for activations: divide by sum of absolute values per token
        #         l1_norm_act = torch.sum(torch.abs(activations), dim=1, keepdim=True) + 1e-12
        #         activations_l1_normalized = torch.abs(activations) / l1_norm_act
                
        #         # L1 norm for max proxy: divide by sum per token
        #         l1_norm_max = torch.sum(torch.abs(last_gen_forward_proxy_max), dim=1, keepdim=True) + 1e-12
        #         max_l1_normalized = last_gen_forward_proxy_max / l1_norm_max
                
        #         # L1 norm for mean proxy: divide by sum per token
        #         l1_norm_mean = torch.sum(torch.abs(last_gen_forward_proxy_mean), dim=1, keepdim=True) + 1e-12
        #         mean_l1_normalized = last_gen_forward_proxy_mean / l1_norm_mean
                
        #         # L2 norm: sqrt(sum of squares) for each neuron across tokens
        #         self.ema_activations[layer_name] = torch.sqrt(
        #             torch.sum(activations_l1_normalized ** 2, dim=0)
        #         )
        #         self.ema_max[layer_name] = torch.sqrt(
        #             torch.sum(max_l1_normalized ** 2, dim=0)
        #         )
        #         self.ema_mean[layer_name] = torch.sqrt(
        #             torch.sum(mean_l1_normalized ** 2, dim=0)
        #         )
        #     else:
        #         # L1 normalize per token before applying exponential weighting
                
        #         # L1 norm for activations
        #         l1_norm_act = torch.sum(torch.abs(activations), dim=1, keepdim=True) + 1e-12
        #         activations_l1_normalized = torch.abs(activations) / l1_norm_act
                
        #         # L1 norm for max proxy
        #         l1_norm_max = torch.sum(torch.abs(last_gen_forward_proxy_max), dim=1, keepdim=True) + 1e-12
        #         max_l1_normalized = last_gen_forward_proxy_max / l1_norm_max
                
        #         # L1 norm for mean proxy
        #         l1_norm_mean = torch.sum(torch.abs(last_gen_forward_proxy_mean), dim=1, keepdim=True) + 1e-12
        #         mean_l1_normalized = last_gen_forward_proxy_mean / l1_norm_mean
                
        #         # Apply exponential weighting to L1-normalized values
        #         weights = self.ema_decay ** torch.arange(seq_length - 1, -1, -1, 
        #                                               dtype=torch.float32, 
        #                                               device=self.device)

        #         weights = weights / weights.sum()
        #         self.ema_activations[layer_name] = torch.sum(
        #             activations_l1_normalized * weights.unsqueeze(1), 
        #             dim=0
        #         )
        #         self.ema_max[layer_name] = torch.sum(
        #             max_l1_normalized * weights.unsqueeze(1), 
        #             dim=0
        #         )
        #         self.ema_mean[layer_name] = torch.sum(
        #             mean_l1_normalized * weights.unsqueeze(1), 
        #             dim=0
        #         )

        # elif self.currIteration <= self.maskingStep: # When currIter == maskingStep, we make the final update and mask, then don't compute ema again.
        #     # Multiply activations with proxy values (broadcasting)
        #     last_gen_forward_proxy_max = torch.abs(activations[-1]) * self.forward_proxies_max[layer_name]
        #     last_gen_forward_proxy_mean = torch.abs(activations[-1]) * self.forward_proxies_mean[layer_name]
            
        #     # Check if ema_decay is None (use L2 norm accumulation)
        #     if self.ema_decay is None:
        #         # L1 normalize the new token's values
        #         l1_norm_max = torch.sum(torch.abs(last_gen_forward_proxy_max)) + 1e-12
        #         max_l1_normalized = last_gen_forward_proxy_max / l1_norm_max
                
        #         l1_norm_mean = torch.sum(torch.abs(last_gen_forward_proxy_mean)) + 1e-12
        #         mean_l1_normalized = last_gen_forward_proxy_mean / l1_norm_mean
                
        #         # L2 norm accumulation: sqrt(sum_old^2 + new_normalized^2)
        #         self.ema_max[layer_name] = torch.sqrt(
        #             self.ema_max[layer_name] ** 2 + max_l1_normalized ** 2
        #         )
        #         self.ema_mean[layer_name] = torch.sqrt(
        #             self.ema_mean[layer_name] ** 2 + mean_l1_normalized ** 2
        #         )
        #     else:
        #         # L1 normalize before EMA update
        #         l1_norm_max = torch.sum(torch.abs(last_gen_forward_proxy_max)) + 1e-12
        #         max_l1_normalized = last_gen_forward_proxy_max / l1_norm_max
                
        #         l1_norm_mean = torch.sum(torch.abs(last_gen_forward_proxy_mean)) + 1e-12
        #         mean_l1_normalized = last_gen_forward_proxy_mean / l1_norm_mean
                
        #         # Update EMA with L1-normalized values: ema_new = decay * ema_old + (1 - decay) * new_normalized
        #         self.ema_max[layer_name] = self.ema_decay * self.ema_max[layer_name] + (1 - self.ema_decay) * max_l1_normalized
        #         self.ema_mean[layer_name] = self.ema_decay * self.ema_mean[layer_name] + (1 - self.ema_decay) * mean_l1_normalized
