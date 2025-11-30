from xml.parsers.expat import model
import torch
import time
import psutil
import threading
import numpy as np
from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import argparse
from importlib.metadata import version

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except:
    HAS_GPU = False

# Print package versions and GPU info
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

@dataclass
class ModelInfo:
    """Model size and parameter information"""
    model_name: str
    total_params: int
    trainable_params: int
    model_size_mb: float
    memory_footprint_mb: float

@dataclass
class TimeSeriesPoint:
    """Single time point measurement"""
    timestamp: float  # seconds since start
    gpu_memory_mb: float
    gpu_utilization: float
    cpu_memory_mb: float
    cpu_utilization: float

@dataclass
class InferenceMetrics:
    model_name: str
    input_tokens: int
    output_tokens: int
    
    # Timing
    total_time_seconds: float
    time_to_first_token_seconds: Optional[float]
    
    # Throughput
    tokens_per_second: float
    output_tokens_per_second: float
    
    # GPU metrics
    peak_gpu_memory_mb: float = 0
    peak_gpu_utilization: float = 0
    
    # Model information
    model_info: Optional[ModelInfo] = None
    
    # Time series data
    time_series: List[TimeSeriesPoint] = None

def get_model_size(model) -> ModelInfo:
    """Calculate comprehensive model size information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in memory (parameters only)
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    # Calculate total memory footprint (including buffers)
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    return ModelInfo(
        model_name=getattr(model, 'name_or_path', 'unknown'),
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=param_size / (1024**2),
        memory_footprint_mb=total_size / (1024**2)
    )

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model

class LLMMetricsCollector:
    def __init__(self, device_id: int = 0, monitoring_interval: float = 0.05):
        self.device_id = device_id
        self.monitoring_interval = monitoring_interval
        
        # Time series monitoring
        self._time_series_data = []
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None
        self._benchmark_start_time = None  # Holding the starting time stamp for benchmark calculations
        self._custom_time = dict()
        
        self.gpu_handle = None
        if HAS_GPU and torch.cuda.is_available():
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                print(f"Warning: Could not initialize GPU {device_id}: {e}")
        
        self._start_gpu_monitoring(time.time())

    def _get_time(self, label: str) -> float:
        """Get elapsed time since start for a custom label"""
        self._custom_time[label] = time.time()

    def _collect_system_metrics(self) -> TimeSeriesPoint:
        """Collect current system metrics at a single point in time"""
        timestamp = time.time() - self._start_time if self._start_time else 0
        
        gpu_memory_mb = 0
        gpu_utilization = 0
        if self.gpu_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_memory_mb = mem_info.used / (1024**2)
                gpu_utilization = util_info.gpu
            except Exception as e:
                # Silent fail for monitoring
                pass
        
        # CPU metrics
        cpu_utilization = psutil.cpu_percent(interval=None)
        cpu_memory_mb = psutil.virtual_memory().used / (1024**2)
        
        return TimeSeriesPoint(
            timestamp=timestamp,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            cpu_memory_mb=cpu_memory_mb,
            cpu_utilization=cpu_utilization
        )

    def _start_gpu_monitoring(self, start_time=None):
        """Start comprehensive system monitoring with time series"""
        self._monitoring = True
        self._time_series_data = []
        self._start_time = start_time if start_time is not None else time.time()
        
        def monitor():
            while self._monitoring:
                try:
                    # Collect comprehensive metrics
                    point = self._collect_system_metrics()
                    self._time_series_data.append(point)
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    # Continue monitoring even if there are errors
                    time.sleep(self.monitoring_interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def _stop_gpu_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _manual_generate(self, model, tokenizer, input_ids, max_new_tokens, temperature=0.7, benchmark_start_time=None):
        """Manual generation with proper KV cache handling"""
        generated_tokens = []
        past_key_values = None
        current_input = input_ids
        
        # Track timing for first token
        first_token_time = None
        # Use benchmark start time if provided, otherwise use current time
        timing_baseline = benchmark_start_time if benchmark_start_time is not None else time.time()
        
        # Collect all generated tokens on GPU first, sync only once at the end
        generated_token_tensors = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(current_input, past_key_values=past_key_values, use_cache=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Update cache (stays on GPU)
                past_key_values = outputs.past_key_values
                
                # Get next token logits (stays on GPU)
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Apply temperature and sample (stays on GPU)
                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    # SYNC POINT: After softmax, before sampling
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                # SYNC POINT: After sampling operation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Record first token time (only CPU timing, no GPU->CPU transfer yet)
                if first_token_time is None:
                    first_token_time = time.time() - timing_baseline
                
                # Keep token on GPU for now, collect all at once later
                generated_token_tensors.append(next_token.clone())
                current_input = next_token.unsqueeze(0)

                # Check for EOS without GPU->CPU transfer (compare tensors on GPU)
                if tokenizer.eos_token_id is not None:
                    eos_tensor = torch.tensor([tokenizer.eos_token_id], device=next_token.device)
                    # SYNC POINT: After tensor creation, before comparison
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if torch.equal(next_token, eos_tensor):
                        break
            
            # SINGLE SYNC: Only synchronize once at the very end
            # This preserves GPU pipeline efficiency while ensuring correctness
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # SYNC POINT: Before any GPU->CPU transfers
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Now convert all tokens to CPU (simple and clear)
        generated_tokens = [token.item() for token in generated_token_tensors]
        
        return input_ids, generated_tokens, first_token_time

    def benchmark(
        self,
        model,
        tokenizer,
        prompt: str,
        model_name: str = "unknown",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **generation_kwargs
    ) -> InferenceMetrics:
        """Benchmark a single inference with comprehensive metrics"""
        
        # Get model size info
        model_info = get_model_size(model)

        # Setup - get device from model instead of assuming
        device = next(model.parameters()).device
        print(f"Model device: {device}")
        
        # Tokenize first
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Ensure model and inputs are fully loaded before benchmarking
        # Sync ensures clean slate before timing starts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Single timing baseline for everything
        start_time = time.time()
        self._benchmark_start_time = start_time  # For time series alignment

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        try:
            
            # Manual forward pass generation only
            final_ids, generated_tokens, first_token_time = self._manual_generate(
                model, tokenizer, inputs.input_ids, max_new_tokens, temperature, start_time
            )
            output_tokens = len(generated_tokens)
            
            total_time = time.time() - start_time
                
        finally:
            self._stop_gpu_monitoring()
        
        # Calculate metrics
        generation_time = total_time - (first_token_time or 0)
        
        # GPU metrics from time series data
        gpu_memory_values = [p.gpu_memory_mb for p in self._time_series_data if p.gpu_memory_mb > 0]
        gpu_util_values = [p.gpu_utilization for p in self._time_series_data]
        
        # Use time series data for peak values
        peak_memory = max(gpu_memory_values) if gpu_memory_values else 0
        peak_util = max(gpu_util_values) if gpu_util_values else 0
        
        metrics = InferenceMetrics(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_time_seconds=total_time,
            time_to_first_token_seconds=first_token_time,
            tokens_per_second=(input_tokens + output_tokens) / total_time,
            output_tokens_per_second=output_tokens / generation_time if generation_time > 0 else 0,
            peak_gpu_memory_mb=peak_memory,
            peak_gpu_utilization=peak_util,
            model_info=model_info,
            time_series=self._time_series_data.copy()  # Copy to avoid reference issues
        )
        
        return metrics
    
    def save_metrics(self, metrics: InferenceMetrics, filename: str = "metrics.json"):
        """Save comprehensive metrics to JSON including time series"""

        # Create comprehensive data structure
        data = {
            'metrics': {**asdict(metrics)},
            'time_series_stats': self._get_time_series_stats(metrics.time_series) if metrics.time_series else {}
        }
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved comprehensive metrics to {filename}")
    
    def _get_time_series_stats(self, time_series: List[TimeSeriesPoint]) -> Dict[str, Any]:
        """Calculate statistics from time series data"""
        if not time_series:
            return {}
        
        gpu_memory = [p.gpu_memory_mb for p in time_series if p.gpu_memory_mb > 0]
        gpu_util = [p.gpu_utilization for p in time_series]
        cpu_memory = [p.cpu_memory_mb for p in time_series]
        cpu_util = [p.cpu_utilization for p in time_series]
        
        stats = {
            'duration_seconds': time_series[-1].timestamp if time_series else 0,
            'data_points': len(time_series),
            'gpu_memory': {
                'min_mb': min(gpu_memory) if gpu_memory else 0,
                'max_mb': max(gpu_memory) if gpu_memory else 0,
                'avg_mb': np.mean(gpu_memory) if gpu_memory else 0,
                'std_mb': np.std(gpu_memory) if gpu_memory else 0
            },
            'gpu_utilization': {
                'min_percent': min(gpu_util) if gpu_util else 0,
                'max_percent': max(gpu_util) if gpu_util else 0,
                'avg_percent': np.mean(gpu_util) if gpu_util else 0,
                'std_percent': np.std(gpu_util) if gpu_util else 0
            },
            'cpu_memory': {
                'min_mb': min(cpu_memory) if cpu_memory else 0,
                'max_mb': max(cpu_memory) if cpu_memory else 0,
                'avg_mb': np.mean(cpu_memory) if cpu_memory else 0,
                'std_mb': np.std(cpu_memory) if cpu_memory else 0
            },
            'cpu_utilization': {
                'min_percent': min(cpu_util) if cpu_util else 0,
                'max_percent': max(cpu_util) if cpu_util else 0,
                'avg_percent': np.mean(cpu_util) if cpu_util else 0,
                'std_percent': np.std(cpu_util) if cpu_util else 0
            }
        }
        
        return stats

    def plot_time_series(self, metrics: InferenceMetrics, save_path: str = None):
        """Plot time series data (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return
        
        if not metrics.time_series:
            print("No time series data available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = [p.timestamp for p in metrics.time_series]
        
        # GPU Memory
        gpu_memory = [p.gpu_memory_mb for p in metrics.time_series]
        ax1.plot(timestamps, gpu_memory, 'b-', linewidth=2)
        ax1.set_title('GPU Memory Usage Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory (MB)')
        ax1.grid(True, alpha=0.3)
        
        # GPU Utilization
        gpu_util = [p.gpu_utilization for p in metrics.time_series]
        ax2.plot(timestamps, gpu_util, 'r-', linewidth=2)
        ax2.set_title('GPU Utilization Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # CPU Memory
        cpu_memory = [p.cpu_memory_mb for p in metrics.time_series]
        ax3.plot(timestamps, cpu_memory, 'g-', linewidth=2)
        ax3.set_title('CPU Memory Usage Over Time')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (MB)')
        ax3.grid(True, alpha=0.3)
        
        # CPU Utilization
        cpu_util = [p.cpu_utilization for p in metrics.time_series]
        ax4.plot(timestamps, cpu_util, 'orange', linewidth=2)
        ax4.set_title('CPU Utilization Over Time')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Utilization (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Add generation phases
        actual_ttft = metrics.time_to_first_token_seconds + (self._benchmark_start_time - self._start_time)
        actual_total_time = metrics.total_time_seconds + (self._benchmark_start_time - self._start_time)
        if metrics.time_to_first_token_seconds:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=(self._benchmark_start_time-self._start_time), color='blue', linestyle='--', alpha=0.7, label='Start Generation')
                ax.axvline(x=actual_ttft, color='red', linestyle='--', alpha=0.7, label='First Token')
                ax.axvline(x=actual_total_time, color='green', linestyle='--', alpha=0.7, label='End Generation')
                for label, time_val in self._custom_time.items():
                    ax.axvline(x = time_val-self._start_time, color='black', linestyle='--', alpha=0.7, label=label)
                ax.legend()
        
        # Title with model info
        title = f'Resource Utilization - {metrics.model_name}'
        if metrics.model_info:
            title += f'\nModel Size: {metrics.model_info.model_size_mb:.1f}MB, Parameters: {metrics.model_info.total_params:,}'
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt2", help='LLM model name or path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    llm_path = os.path.join(os.path.dirname(__file__), "..", "llm_weights")
    parser.add_argument("--cache_dir", default=llm_path, type=str)
    save_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    parser.add_argument('--save_dir', type=str, default=save_dir, help='Path to save logs')
    parser.add_argument('--save_result', type=str, default=None, help='The results to save(all, activations, activation_plots, contribution_plots, analysis)')
    parser.add_argument('--save_model_dir', type=str, default=None, help='Path to save the model')
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Maximum number of new tokens to generate')
    parser.add_argument('--plot', action='store_true', help='Generate time series plots')
    parser.add_argument('--monitoring_interval', type=float, default=0.05, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    max_new_tokens = args.max_new_tokens  # Maximum new tokens to generate
    
    collector = LLMMetricsCollector(monitoring_interval = args.monitoring_interval)  # High frequency monitoring
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ########### ---- Load tokenizer
    collector._get_time(label="load_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize and run prompt
    #prompt_imc = "In Memory Computing is an emerging paradigm that integrates memory and computation to enhance performance and efficiency in computing systems."
    #prompt_bird = "The bird flew over the ocean. Drums make loud sounds. Pizzas are a popular food item in New York and Chicago. One should break"
    #prompt_capital = "The capital of France is Paris. The capital of Germany is Berlin. The capital of India is Delhi. The best tourist destination is"
    #prompt_pizza = "Pizza is a globally popular food found across the world, with billions sold annually, but there isn't a single 'pizza in the world'; instead, there are countless varieties like the classic Neapolitan, New York-style, Chicago deep-dish, and international adaptations such as Japanese okonomiyaki or Tanzanian Zanzibar pizza, alongside numerous high-ranking pizzerias worldwide."
    prompt_adam = "Adam (Adaptive Moment Estimation) is a widely used optimization algorithm in deep learning, known for its efficiency and effectiveness in training neural networks. It combines the advantages of two other popular optimization methods: RMSProp and Momentum."
    prompt = prompt_adam

    ########### ---- Load model
    print(f"Loading model: {args.model}")
    collector._get_time(label="load_model_start")
    model = get_llm(args.model, args.cache_dir)
    collector._get_time(label="load_model_end")
    model.eval()

    # Run benchmark with manual generation (forward passes only)
    metrics = collector.benchmark(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        model_name=args.model,
        max_new_tokens=max_new_tokens,
        temperature=0.0
    )
    
    # Save metrics with time series
    os.makedirs(args.save_dir, exist_ok=True)
    safe_model_name = args.model.replace('/', '_')
    metrics_file = os.path.join(args.save_dir, f"metrics_{safe_model_name}.json")
    collector.save_metrics(metrics, metrics_file)
    
    # Generate plots if requested
    if args.plot:
        plot_file = os.path.join(args.save_dir, f"plot_{safe_model_name}.png")
        collector.plot_time_series(metrics, plot_file)

if __name__ == "__main__":
    main()