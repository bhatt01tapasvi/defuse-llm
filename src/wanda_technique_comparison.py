import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
import psutil
import os

@dataclass
class ComputationMetrics:
    """Store metrics for a pruning method"""
    method_name: str
    flops: int  # Floating point operations
    memory_peak_mb: float
    memory_allocated_mb: float
    time_seconds: float
    num_matmuls: int
    num_elementwise_ops: int
    
class PruningProfiler:
    """Profile and compare pruning methods"""
    
    def __init__(self, hidden_dim=4096, vocab_size=32000, num_layers=32, 
                 seq_len=128, nsamples=128):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.nsamples = nsamples
        
    def count_flops(self, operation: str, *shapes) -> int:
        """Count FLOPs for common operations"""
        if operation == "matmul":
            # C = A @ B where A is (m, k) and B is (k, n)
            m, k = shapes[0]
            _, n = shapes[1]
            return 2 * m * k * n  # multiply-add = 2 ops
        elif operation == "elementwise":
            # Element-wise ops (multiply, add, abs, sqrt, etc.)
            return np.prod(shapes[0])
        elif operation == "reduction":
            # Sum, mean, etc.
            return np.prod(shapes[0])
        return 0
    
    def profile_wanda_style(self, device='cuda'):
        """
        Profile Wanda-style pruning approach.
        
        Wanda Algorithm:
        1. For each layer, accumulate activation statistics during forward passes
        2. Compute W_metric = |W| * sqrt(activation_variance) for each neuron
        3. Sort and prune based on W_metric
        
        Key operations per layer:
        - Accumulate activations²: nsamples × seq_len × hidden_dim elementwise ops
        - Compute metric: hidden_dim² elementwise ops (abs, multiply, sqrt)
        - Sort: O(hidden_dim² log(hidden_dim))
        """
        print("\n" + "="*80)
        print("Profiling Wanda-Style Pruning")
        print("="*80)
        
        metrics = ComputationMetrics(
            method_name="Wanda",
            flops=0,
            memory_peak_mb=0,
            memory_allocated_mb=0,
            time_seconds=0,
            num_matmuls=0,
            num_elementwise_ops=0
        )
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_time = time.time()
        
        # Wanda processes layer by layer during calibration
        for layer_idx in range(self.num_layers):
            # Simulate weight matrix (hidden_dim × hidden_dim)
            W = torch.randn(self.hidden_dim, self.hidden_dim, device=device)
            
            # Accumulate activation statistics (per neuron)
            # Shape: (hidden_dim,)
            scaler_row = torch.zeros(self.hidden_dim, device=device)
            
            # Process all calibration samples
            for sample_idx in range(self.nsamples):
                # Simulate activations for this sample
                # Shape: (seq_len, hidden_dim)
                activations = torch.randn(self.seq_len, self.hidden_dim, device=device)
                
                # Accumulate squared activations across sequence
                # scaler_row += sum(activations²) over seq_len
                scaler_row += torch.sum(activations ** 2, dim=0)
                
                # Count operations:
                # - Square: seq_len × hidden_dim ops
                # - Sum reduction: seq_len × hidden_dim ops
                metrics.flops += self.count_flops("elementwise", (self.seq_len, self.hidden_dim))
                metrics.flops += self.count_flops("reduction", (self.seq_len, self.hidden_dim))
                metrics.num_elementwise_ops += 2
            
            # Finalize scaler: sqrt(sum / nsamples)
            scaler_row = torch.sqrt(scaler_row / self.nsamples)
            metrics.flops += self.count_flops("elementwise", (self.hidden_dim,)) * 2  # divide + sqrt
            metrics.num_elementwise_ops += 2
            
            # Compute W_metric = |W| * sqrt(scaler_row)
            # Shape: (hidden_dim, hidden_dim)
            W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape((1, -1)))
            metrics.flops += self.count_flops("elementwise", (self.hidden_dim, self.hidden_dim)) * 3  # abs + sqrt + multiply
            metrics.num_elementwise_ops += 3
            
            # Sorting for pruning (O(n log n) where n = hidden_dim²)
            total_weights = self.hidden_dim * self.hidden_dim
            sort_flops = total_weights * np.log2(total_weights) * 2
            metrics.flops += int(sort_flops)
            
            # Apply pruning mask (zero out weights)
            prune_ratio = 0.5
            num_pruned = int(total_weights * prune_ratio)
            
            del W, scaler_row, W_metric
            torch.cuda.empty_cache()
        
        metrics.time_seconds = time.time() - start_time
        metrics.memory_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024**2
        
        print(f"Total FLOPs: {metrics.flops:,}")
        print(f"Time: {metrics.time_seconds:.2f}s")
        print(f"Peak Memory: {metrics.memory_peak_mb:.2f} MB")
        
        return metrics
    
    def profile_neuron_defuser_magnitude(self, device='cuda', ema_decay=None):
        """
        Profile NeuronDefuser approach using MAGNITUDE-ONLY scoring.
        This matches your 'magnitude' ranking method where you simply accumulate
        activation magnitudes over time.
        
        NeuronDefuser Algorithm (Magnitude-only):
        1. During generation, accumulate |activations| per neuron over time
        2. Use either:
           - L2 norm: sqrt(sum(|activation|²)) across all tokens
           - EMA: exponentially weighted average of |activations|
        3. At maskingStep, rank neurons by accumulated magnitude
        4. Keep top-k neurons per layer
        
        Key differences from Wanda:
        - No weight matrix involvement (just activation magnitudes)
        - Accumulates incrementally during generation (token by token after first iteration)
        - Deferred pruning decision (all accumulation first, then prune)
        """
        print("\n" + "="*80)
        print(f"Profiling NeuronDefuser Magnitude-Only (ema_decay={ema_decay})")
        print("="*80)
        
        metrics = ComputationMetrics(
            method_name=f"NeuronDefuser-Mag (ema={ema_decay})",
            flops=0,
            memory_peak_mb=0,
            memory_allocated_mb=0,
            time_seconds=0,
            num_matmuls=0,
            num_elementwise_ops=0
        )
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_time = time.time()
        
        # Initialize EMA storage for all layers
        # Shape per layer: (hidden_dim,)
        ema_activations = [torch.zeros(self.hidden_dim, device=device) 
                          for _ in range(self.num_layers)]
        
        print("Accumulating activation magnitudes during generation...")
        
        # Accumulate during generation (token by token)
        for sample_idx in range(self.nsamples):
            for layer_idx in range(self.num_layers):
                # Simulate activations for this token/sample
                # Shape: (seq_len, hidden_dim)
                activations = torch.randn(self.seq_len, self.hidden_dim, device=device)
                
                if ema_decay is None:
                    # L2 norm accumulation: sum of squares
                    if sample_idx == 0:
                        # First iteration: process all tokens in sequence
                        # ema = sum(|activations|²) over seq_len
                        ema_activations[layer_idx] = torch.sum(
                            torch.abs(activations) ** 2, dim=0
                        )
                        
                        # Count ops:
                        # - abs: seq_len × hidden_dim
                        # - square: seq_len × hidden_dim  
                        # - sum: seq_len × hidden_dim
                        metrics.flops += self.count_flops("elementwise", activations.shape) * 2  # abs + square
                        metrics.flops += self.count_flops("reduction", activations.shape)  # sum
                        metrics.num_elementwise_ops += 3
                    else:
                        # Subsequent iterations: only process last token
                        # ema += |last_token|²
                        last_act = torch.abs(activations[-1])
                        ema_activations[layer_idx] += last_act ** 2
                        
                        # Count ops per neuron:
                        # - abs: hidden_dim
                        # - square: hidden_dim
                        # - add: hidden_dim
                        metrics.flops += self.count_flops("elementwise", (self.hidden_dim,)) * 3
                        metrics.num_elementwise_ops += 3
                
                else:
                    # EMA accumulation: exponentially weighted average
                    if sample_idx == 0:
                        # First iteration: weighted sum over sequence
                        # weights = decay^(seq_len-1-i) for i in range(seq_len)
                        weights = ema_decay ** torch.arange(self.seq_len - 1, -1, -1, 
                                                           dtype=torch.float32, device=device)
                        weights = weights / weights.sum()
                        
                        # ema = sum(|activations| * weights)
                        ema_activations[layer_idx] = torch.sum(
                            torch.abs(activations) * weights.unsqueeze(1), dim=0
                        )
                        
                        # Count ops:
                        # - abs: seq_len × hidden_dim
                        # - multiply (weights): seq_len × hidden_dim
                        # - sum: seq_len × hidden_dim
                        metrics.flops += self.count_flops("elementwise", activations.shape) * 2  # abs + multiply
                        metrics.flops += self.count_flops("reduction", activations.shape)  # sum
                        metrics.num_elementwise_ops += 3
                    else:
                        # Subsequent iterations: EMA update with last token
                        # ema = decay × ema + (1 - decay) × |last_token|
                        last_act = torch.abs(activations[-1])
                        ema_activations[layer_idx] = (
                            ema_decay * ema_activations[layer_idx] + 
                            (1 - ema_decay) * last_act
                        )
                        
                        # Count ops per neuron:
                        # - abs: hidden_dim
                        # - multiply (×2): hidden_dim × 2
                        # - add: hidden_dim
                        metrics.flops += self.count_flops("elementwise", (self.hidden_dim,)) * 4
                        metrics.num_elementwise_ops += 4
        
        # Final scoring and pruning (happens at maskingStep)
        print("Computing final scores and pruning...")
        for layer_idx in range(self.num_layers):
            if ema_decay is None:
                # Take sqrt to get actual L2 norm
                act_score = torch.sqrt(ema_activations[layer_idx])
                metrics.flops += self.count_flops("elementwise", (self.hidden_dim,))
                metrics.num_elementwise_ops += 1
            else:
                # EMA score is already in correct form
                act_score = ema_activations[layer_idx]
            
            # Top-k selection (keep 50% of neurons)
            k = int(self.hidden_dim * 0.5)
            topk_values, topk_indices = torch.topk(act_score, k)
            
            # Approximate top-k as O(n log k)
            topk_flops = self.hidden_dim * np.log2(k)
            metrics.flops += int(topk_flops)
            
            # Create and apply mask (not counted in FLOPs as it's a one-time cost)
        
        metrics.time_seconds = time.time() - start_time
        metrics.memory_peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024**2
        
        print(f"Total FLOPs: {metrics.flops:,}")
        print(f"Number of matmuls: {metrics.num_matmuls}")
        print(f"Time: {metrics.time_seconds:.2f}s")
        print(f"Peak Memory: {metrics.memory_peak_mb:.2f} MB")
        
        return metrics
    
    def compare_and_plot(self, results: List[ComputationMetrics], save_path='comparison_plots.png'):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Pruning Methods Comparison: Wanda vs NeuronDefuser (Magnitude-Only)', 
                    fontsize=16, fontweight='bold')
        
        methods = [r.method_name for r in results]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(results)]
        
        # 1. FLOPs comparison
        ax = axes[0, 0]
        flops = [r.flops for r in results]
        bars = ax.bar(methods, flops, color=colors)
        ax.set_ylabel('FLOPs', fontsize=11, fontweight='bold')
        ax.set_title('Total Floating Point Operations', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, flops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val/1e9:.2f}G', ha='center', va='bottom', fontsize=10)
        
        # 2. Time comparison
        ax = axes[0, 1]
        times = [r.time_seconds for r in results]
        bars = ax.bar(methods, times, color=colors)
        ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Execution Time', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # 3. Memory comparison
        ax = axes[0, 2]
        memory = [r.memory_peak_mb for r in results]
        bars = ax.bar(methods, memory, color=colors)
        ax.set_ylabel('Memory (MB)', fontsize=11, fontweight='bold')
        ax.set_title('Peak Memory Usage', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, memory):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}MB', ha='center', va='bottom', fontsize=10)
        
        # 4. FLOPs breakdown
        ax = axes[1, 0]
        elementwise = [r.num_elementwise_ops for r in results]
        bars = ax.bar(methods, elementwise, color=colors)
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Number of Elementwise Operations', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, elementwise):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontsize=10)
        
        # 5. Relative speedup
        ax = axes[1, 1]
        baseline_time = results[0].time_seconds
        speedups = [baseline_time / r.time_seconds for r in results]
        bars = ax.bar(methods, speedups, color=colors)
        ax.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
        ax.set_title(f'Relative Speedup (vs {methods[0]})', fontsize=12, fontweight='bold')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Baseline')
        ax.tick_params(axis='x', rotation=15)
        ax.legend()
        for bar, val in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # 6. FLOPs efficiency (FLOPs per second)
        ax = axes[1, 2]
        efficiency = [r.flops / r.time_seconds / 1e9 for r in results]  # GFLOPs/s
        bars = ax.bar(methods, efficiency, color=colors)
        ax.set_ylabel('GFLOPs/s', fontsize=11, fontweight='bold')
        ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {save_path}")
        plt.close()
        
        # Print detailed comparison table
        print("\n" + "="*110)
        print("DETAILED COMPARISON TABLE")
        print("="*110)
        print(f"{'Method':<30} {'FLOPs':<15} {'Time (s)':<12} {'Memory (MB)':<15} {'Elem Ops':<12} {'Speedup':<10}")
        print("-"*110)
        
        baseline_time = results[0].time_seconds
        for r in results:
            speedup = baseline_time / r.time_seconds
            print(f"{r.method_name:<30} {r.flops/1e9:>10.2f}G {r.time_seconds:>10.2f} "
                  f"{r.memory_peak_mb:>13.2f} {r.num_elementwise_ops:>10} {speedup:>8.2f}x")
        print("="*110)
        
        # Print algorithmic insights
        print("\n" + "="*110)
        print("ALGORITHMIC INSIGHTS")
        print("="*110)
        print("\nWanda Algorithm:")
        print("  - Accumulates activation² for each neuron during calibration")
        print("  - Computes importance = |Weight| × sqrt(activation_variance)")
        print("  - Requires weight matrix access during scoring")
        print("  - Prunes layer-by-layer during calibration")
        print(f"  - Total ops per layer: O(nsamples × seq_len × hidden_dim)")
        
        print("\nNeuronDefuser (Magnitude-Only) Algorithm:")
        print("  - Accumulates |activation| magnitudes incrementally")
        print("  - No weight matrix access needed for scoring")
        print("  - Deferred pruning (accumulate first, then prune)")
        print("  - First iteration: process full sequence")
        print("  - Subsequent: only process last token (incremental)")
        print(f"  - Total ops per layer: O(seq_len × hidden_dim + nsamples × hidden_dim)")
        
        # Calculate reduction ratios
        wanda_flops = results[0].flops
        defuser_flops = results[1].flops if len(results) > 1 else 0
        if defuser_flops > 0:
            reduction = ((wanda_flops - defuser_flops) / wanda_flops) * 100
            print(f"\nFLOPs Reduction: {reduction:.1f}% fewer operations with NeuronDefuser")
        print("="*110)


# Example usage
if __name__ == "__main__":
    # Initialize profiler with model specs (e.g., LLaMA-7B)
    profiler = PruningProfiler(
        hidden_dim=4096,     # MLP hidden dimension
        vocab_size=32000,    # Not used in magnitude-only approach
        num_layers=32,       # Number of transformer layers
        seq_len=128,         # Sequence length for calibration
        nsamples=128         # Number of calibration samples
    )
    
    # Profile both methods
    results = []
    
    print("\n" + "="*80)
    print("COMPARING WANDA VS NEURONDEFUSER (MAGNITUDE-ONLY SCORING)")
    print("="*80)
    
    # Wanda baseline
    results.append(profiler.profile_wanda_style())
    
    # NeuronDefuser with L2 norm (no EMA)
    results.append(profiler.profile_neuron_defuser_magnitude(ema_decay=None))
    
    # NeuronDefuser with EMA
    results.append(profiler.profile_neuron_defuser_magnitude(ema_decay=0.9))
    
    # Create comparison plots
    profiler.compare_and_plot(results, save_path='pruning_comparison_magnitude.png')