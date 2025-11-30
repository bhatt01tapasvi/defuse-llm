import torch
import os
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRUNE_DIR = os.path.join(RESULTS_DIR, "pruneNeurons")
#MAX_FORWARD_PROXY = os.path.join(PRUNING_DIR, "maxProxy")

class NeuronDefuser:
    def __init__(self, maskingStep: int=0, per_layer_topk: dict=None, ema_decay: float=0.5, ranking_method: str='combined', device: str='cuda'):
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

        # Store the original per_layer_topk config
        self.per_layer_topk_config = per_layer_topk if per_layer_topk is not None else {}

        self.per_layer_topk = {}
        self.layer_name_to_number = {}  # Map layer names to their numbers
        self.layer_number_to_name = {}  # Map layer numbers to their names

        self.forward_proxies_max = {}  # Store max forward proxies per layer
        self.forward_proxies_mean = {}  # Store mean forward proxies per layer
        self.ema_max = {}  # Store EMA per layer
        self.ema_mean = {}  # Store EMA per layer
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

        # Calculate forward proxy and keep it on GPU
        forward_proxy = (weight @ embedding_weights.T).detach()  # Remove .cpu().numpy()
        self.forward_proxies_max[layer_name] = torch.max(torch.abs(forward_proxy), dim=1)[0]
        self.forward_proxies_mean[layer_name] = torch.mean(torch.abs(forward_proxy), dim=1)

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def _compute_combined_score(self, max_scores: torch.Tensor, 
                                mean_scores: torch.Tensor) -> torch.Tensor:
        if self.ranking_method == 'max':
            return max_scores
        elif self.ranking_method == 'mean':
            return mean_scores
        elif self.ranking_method == 'combined':
            # Normalize both scores and take weighted sum
            norm_max = self._normalize_scores(max_scores)
            norm_mean = self._normalize_scores(mean_scores)
            return norm_max + norm_mean  # Equal weighting
        elif self.ranking_method == 'product':
            # Multiply normalized scores (favors neurons high in both)
            norm_max = self._normalize_scores(max_scores)
            norm_mean = self._normalize_scores(mean_scores)
            return norm_max * norm_mean
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
        
        batch_size, seq_len, hidden_dim = activations3.shape
        activations = activations3[0]

        total_neurons = hidden_dim

        if self.currIteration == 0:
            # Multiply activations with proxy values (broadcasting)
            last_gen_forward_proxy_max = torch.abs(activations) * self.forward_proxies_max[layer_name]
            last_gen_forward_proxy_mean = torch.abs(activations) * self.forward_proxies_mean[layer_name]
            
            # Update the EMA
            # Initialize EMA on first call
            for token_idx in range(last_gen_forward_proxy_max.shape[0]):
                # I am skipping the if approach for now because the first token usually is a pretty
                if token_idx == 0:
                    self.ema_max[layer_name] = (1 - self.ema_decay) * last_gen_forward_proxy_max[token_idx].clone()
                    self.ema_mean[layer_name] = (1 - self.ema_decay) * last_gen_forward_proxy_mean[token_idx].clone()
                else:
                    self.ema_max[layer_name] = self.ema_decay * self.ema_max[layer_name] + (1 - self.ema_decay) * last_gen_forward_proxy_max[token_idx]
                    self.ema_mean[layer_name] = self.ema_decay * self.ema_mean[layer_name] + (1 - self.ema_decay) * last_gen_forward_proxy_mean[token_idx]
        
        elif self.currIteration <= self.maskingStep: # When currIter == maskingStep, we make the final update and mask, then don't compute ema again.
            # Multiply activations with proxy values (broadcasting)
            last_gen_forward_proxy_max = torch.abs(activations[-1]) * self.forward_proxies_max[layer_name]  # Shape: (num_tokens, embed_dim)
            last_gen_forward_proxy_mean = torch.abs(activations[-1]) * self.forward_proxies_mean[layer_name]
            
            # Update EMA: ema_new = decay * ema_old + (1 - decay) * new_value
            self.ema_max[layer_name] = self.ema_decay * self.ema_max[layer_name] + (1 - self.ema_decay) * last_gen_forward_proxy_max
            self.ema_mean[layer_name] = self.ema_decay * self.ema_mean[layer_name] + (1 - self.ema_decay) * last_gen_forward_proxy_mean

        # Keep a count of the iteration we are at.
        if self.currIteration < self.maskingStep:
            if layer_name == list(self.forward_proxies_max.keys())[-1]:
                self.currIteration += 1
            return activations3
        elif self.currIteration == self.maskingStep:
            # Check if this layer should be pruned
            layer_topk = self.per_layer_topk.get(layer_name, -1)
            layer_num = self.layer_name_to_number.get(layer_name, -1)
            
            if layer_topk == -1:
                # Don't prune this layer - keep all neurons
                print(f"Skipping pruning for layer {layer_num}: {layer_name}")
                self.masks[layer_name] = None  # No mask means keep all
            else:
                # Prune this layer
                combined_score = self._compute_combined_score(
                    self.ema_max[layer_name], 
                    self.ema_mean[layer_name]
                )
                
                # Get top-K neurons based on combined ranking
                topk_values, topk_indices = torch.topk(combined_score, layer_topk)
                
                # Create mask: keep only top-K neurons
                mask = torch.zeros(total_neurons, dtype=torch.float32, device=self.device)
                mask[topk_indices] = 1.0
                self.masks[layer_name] = mask
                
                # Store detailed statistics
                self.neuron_stats[layer_name] = {
                    'layer_number': layer_num,
                    'total_neurons': total_neurons,
                    'neurons_kept': layer_topk,
                    'neurons_pruned': total_neurons - layer_topk,
                    'pruning_percentage': ((total_neurons - layer_topk) / total_neurons) * 100,
                }
                
        # Increment iteration counter
        if layer_name == list(self.forward_proxies_max.keys())[-1]:
            self.currIteration += 1

        # Apply mask to activations (if mask exists for this layer)
        if layer_name in self.masks and self.masks[layer_name] is not None:
            activations3 *= self.masks[layer_name]
        
        return activations3
    
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

        for layer_name, stats in self.neuron_stats.items():
            # Create layer summary without full score arrays
            summary_data['layers'][layer_name] = {
                'layer_number': stats['layer_number'],
                'total_neurons': stats['total_neurons'],
                'neurons_kept': stats['neurons_kept'],
                'neurons_pruned': stats['neurons_pruned'],
                'pruning_percentage': stats['pruning_percentage'],
            }
        
        # Save summary JSON
        json_file = os.path.join(output_dir, f"{filename_prefix}_summary.json")
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Neuron ranking results saved:")
        print(f"  Summary: {json_file}")
        print(f"{'='*80}\n")
