import torch
import psutil
import os
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
import gc

class MemoryProfiler:
    """
    Utility class to profile CPU and GPU memory usage with continuous monitoring.
    Captures peak memory usage during execution, not just at boundaries.
    
    Usage:
        profiler = MemoryProfiler()
        profiler.start("model_loading")
        # ... your code ...
        profiler.end("model_loading")
        profiler.print_report()
    """
    
    def __init__(self, device='cuda', sample_interval=0.1):
        """
        Args:
            device: CUDA device to monitor
            sample_interval: How often to sample memory (in seconds)
        """
        self.device = device
        self.sample_interval = sample_interval
        self.measurements = defaultdict(dict)
        self.active_sections = {}
        self.process = psutil.Process(os.getpid())
        self.monitoring_threads = {}
        self.stop_monitoring = {}
        
    def get_memory_stats(self):
        """Get current memory statistics for CPU and GPU."""
        stats = {}
        
        # CPU Memory (in MB)
        mem_info = self.process.memory_info()
        stats['cpu_rss_mb'] = mem_info.rss / 1024**2  # Resident Set Size
        stats['cpu_vms_mb'] = mem_info.vms / 1024**2  # Virtual Memory Size
        stats['cpu_percent'] = self.process.memory_percent()
        
        # GPU Memory (in MB)
        if torch.cuda.is_available():
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / 1024**2
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / 1024**2
            stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / 1024**2
            stats['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved(self.device) / 1024**2
        else:
            stats['gpu_allocated_mb'] = 0
            stats['gpu_reserved_mb'] = 0
            stats['gpu_max_allocated_mb'] = 0
            stats['gpu_max_reserved_mb'] = 0
            
        return stats
    
    def _monitor_memory(self, section_name):
        """Background thread that continuously monitors memory usage."""
        peak_stats = {
            'cpu_rss_mb': 0,
            'cpu_vms_mb': 0,
            'gpu_allocated_mb': 0,
            'gpu_reserved_mb': 0,
        }
        
        sample_count = 0
        sum_stats = {
            'cpu_rss_mb': 0,
            'cpu_vms_mb': 0,
            'gpu_allocated_mb': 0,
            'gpu_reserved_mb': 0,
        }
        
        while not self.stop_monitoring.get(section_name, False):
            current_stats = self.get_memory_stats()
            
            # Update peaks
            peak_stats['cpu_rss_mb'] = max(peak_stats['cpu_rss_mb'], current_stats['cpu_rss_mb'])
            peak_stats['cpu_vms_mb'] = max(peak_stats['cpu_vms_mb'], current_stats['cpu_vms_mb'])
            peak_stats['gpu_allocated_mb'] = max(peak_stats['gpu_allocated_mb'], current_stats['gpu_allocated_mb'])
            peak_stats['gpu_reserved_mb'] = max(peak_stats['gpu_reserved_mb'], current_stats['gpu_reserved_mb'])
            
            # Update running averages
            sum_stats['cpu_rss_mb'] += current_stats['cpu_rss_mb']
            sum_stats['cpu_vms_mb'] += current_stats['cpu_vms_mb']
            sum_stats['gpu_allocated_mb'] += current_stats['gpu_allocated_mb']
            sum_stats['gpu_reserved_mb'] += current_stats['gpu_reserved_mb']
            sample_count += 1
            
            time.sleep(self.sample_interval)
        
        # Calculate averages
        avg_stats = {k: v / sample_count if sample_count > 0 else 0 
                     for k, v in sum_stats.items()}
        
        return peak_stats, avg_stats, sample_count
    
    def start(self, section_name):
        """Start profiling a code section with continuous monitoring."""
        if section_name in self.active_sections:
            print(f"Warning: Section '{section_name}' already started")
            return
        
        # Force garbage collection before measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start_stats = self.get_memory_stats()
        
        # Start background monitoring thread
        self.stop_monitoring[section_name] = False
        monitor_thread = threading.Thread(
            target=lambda: self._store_monitoring_results(
                section_name, 
                self._monitor_memory(section_name)
            ),
            daemon=True
        )
        monitor_thread.start()
        
        self.active_sections[section_name] = {
            'start_time': time.time(),
            'start_stats': start_stats,
            'monitor_thread': monitor_thread
        }
    
    def _store_monitoring_results(self, section_name, results):
        """Store monitoring results (called by background thread)."""
        peak_stats, avg_stats, sample_count = results
        if section_name in self.active_sections:
            self.active_sections[section_name]['peak_stats'] = peak_stats
            self.active_sections[section_name]['avg_stats'] = avg_stats
            self.active_sections[section_name]['sample_count'] = sample_count
    
    def end(self, section_name):
        """End profiling a code section."""
        if section_name not in self.active_sections:
            print(f"Warning: Section '{section_name}' was not started")
            return
        
        # Stop monitoring thread
        self.stop_monitoring[section_name] = True
        start_data = self.active_sections[section_name]
        start_data['monitor_thread'].join(timeout=2.0)  # Wait for thread to finish
        
        # Force garbage collection after measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        end_stats = self.get_memory_stats()
        start_stats = start_data['start_stats']
        peak_stats = start_data.get('peak_stats', end_stats)
        avg_stats = start_data.get('avg_stats', end_stats)
        sample_count = start_data.get('sample_count', 0)
        
        # Calculate deltas and peaks
        self.measurements[section_name] = {
            'duration_sec': time.time() - start_data['start_time'],
            'samples_taken': sample_count,
            
            # Start values
            'start_cpu_mb': start_stats['cpu_rss_mb'],
            'start_gpu_mb': start_stats['gpu_allocated_mb'],
            
            # End values
            'end_cpu_mb': end_stats['cpu_rss_mb'],
            'end_gpu_mb': end_stats['gpu_allocated_mb'],
            
            # Deltas (end - start)
            'cpu_rss_delta_mb': end_stats['cpu_rss_mb'] - start_stats['cpu_rss_mb'],
            'cpu_vms_delta_mb': end_stats['cpu_vms_mb'] - start_stats['cpu_vms_mb'],
            'gpu_allocated_delta_mb': end_stats['gpu_allocated_mb'] - start_stats['gpu_allocated_mb'],
            'gpu_reserved_delta_mb': end_stats['gpu_reserved_mb'] - start_stats['gpu_reserved_mb'],
            
            # Peak values (maximum observed during execution)
            'peak_cpu_mb': peak_stats['cpu_rss_mb'],
            'peak_gpu_mb': peak_stats['gpu_allocated_mb'],
            'peak_gpu_reserved_mb': peak_stats['gpu_reserved_mb'],
            
            # Average values (during execution)
            'avg_cpu_mb': avg_stats['cpu_rss_mb'],
            'avg_gpu_mb': avg_stats['gpu_allocated_mb'],
            
            # Peak delta (how much above start was the peak)
            'peak_cpu_delta_mb': peak_stats['cpu_rss_mb'] - start_stats['cpu_rss_mb'],
            'peak_gpu_delta_mb': peak_stats['gpu_allocated_mb'] - start_stats['gpu_allocated_mb'],
        }
        
        del self.active_sections[section_name]
    
    @contextmanager
    def profile(self, section_name):
        """Context manager for profiling a code block."""
        self.start(section_name)
        try:
            yield
        finally:
            self.end(section_name)
    
    def snapshot(self, label="snapshot"):
        """Take a memory snapshot at a specific point."""
        stats = self.get_memory_stats()
        print(f"\n{'='*80}")
        print(f"Memory Snapshot: {label}")
        print(f"{'='*80}")
        print(f"CPU Memory (RSS): {stats['cpu_rss_mb']:>10.1f} MB")
        print(f"CPU Memory (VMS): {stats['cpu_vms_mb']:>10.1f} MB")
        print(f"GPU Allocated:    {stats['gpu_allocated_mb']:>10.1f} MB")
        print(f"GPU Reserved:     {stats['gpu_reserved_mb']:>10.1f} MB")
        print(f"{'='*80}\n")
    
    def print_report(self, show_details=True):
        """Print a comprehensive memory usage report."""
        if not self.measurements:
            print("No measurements recorded")
            return
        
        print("\n" + "="*120)
        print("MEMORY PROFILING REPORT (with Peak Monitoring)")
        print("="*120)
        print(f"{'Section':<25} {'Duration':<10} {'Samples':<10} {'CPU Δ':<12} {'GPU Δ':<12} {'Peak CPU Δ':<12} {'Peak GPU Δ':<12}")
        print(f"{'':25} {'(sec)':<10} {'':10} {'(MB)':<12} {'(MB)':<12} {'(MB)':<12} {'(MB)':<12}")
        print("-"*120)
        
        for section, data in self.measurements.items():
            print(f"{section:<25} "
                  f"{data['duration_sec']:>8.3f}  "
                  f"{data['samples_taken']:>8d}  "
                  f"{data['cpu_rss_delta_mb']:>+10.1f}  "
                  f"{data['gpu_allocated_delta_mb']:>+10.1f}  "
                  f"{data['peak_cpu_delta_mb']:>+10.1f}  "
                  f"{data['peak_gpu_delta_mb']:>+10.1f}")
        
        print("="*120)
        
        if show_details:
            # Detailed breakdown
            print("\nDETAILED BREAKDOWN:")
            print("-"*120)
            for section, data in self.measurements.items():
                print(f"\n{section}:")
                print(f"  Duration:              {data['duration_sec']:.3f} sec")
                print(f"  Samples taken:         {data['samples_taken']}")
                print(f"  CPU Memory:")
                print(f"    Start:               {data['start_cpu_mb']:>10.1f} MB")
                print(f"    End:                 {data['end_cpu_mb']:>10.1f} MB")
                print(f"    Peak:                {data['peak_cpu_mb']:>10.1f} MB  ⭐")
                print(f"    Average:             {data['avg_cpu_mb']:>10.1f} MB")
                print(f"    Delta (start→end):   {data['cpu_rss_delta_mb']:>+10.1f} MB")
                print(f"    Peak Delta:          {data['peak_cpu_delta_mb']:>+10.1f} MB  ⭐")
                print(f"  GPU Memory:")
                print(f"    Start:               {data['start_gpu_mb']:>10.1f} MB")
                print(f"    End:                 {data['end_gpu_mb']:>10.1f} MB")
                print(f"    Peak:                {data['peak_gpu_mb']:>10.1f} MB  ⭐")
                print(f"    Average:             {data['avg_gpu_mb']:>10.1f} MB")
                print(f"    Delta (start→end):   {data['gpu_allocated_delta_mb']:>+10.1f} MB")
                print(f"    Peak Delta:          {data['peak_gpu_delta_mb']:>+10.1f} MB  ⭐")
            
            print("\n⭐ = Peak values show maximum memory usage during execution")
            print("-"*120)
        
        # Summary of top memory consumers
        print("\nTOP MEMORY CONSUMERS (by Peak GPU Delta):")
        print("-"*120)
        top_sections = sorted(
            self.measurements.items(),
            key=lambda x: x[1]['peak_gpu_delta_mb'],
            reverse=True
        )[:5]
        
        for i, (section, data) in enumerate(top_sections, 1):
            print(f"{i}. {section:<30} Peak GPU: +{data['peak_gpu_delta_mb']:>10.1f} MB "
                  f"(avg: {data['avg_gpu_mb']:>10.1f} MB)")
        
        print("="*120 + "\n")
    
    def get_top_memory_sections(self, n=5, metric='peak_gpu_delta_mb'):
        """Get top N sections by memory usage."""
        sorted_sections = sorted(
            self.measurements.items(),
            key=lambda x: abs(x[1].get(metric, 0)),
            reverse=True
        )
        return sorted_sections[:n]
    
    def reset_peak_stats(self):
        """Reset peak memory statistics (useful between runs)."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def save_report(self, filepath):
        """Save the profiling report to a file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.measurements, f, indent=2)
        print(f"Memory profiling report saved to: {filepath}")


def profile_tensor_memory(name="Tensor"):
    """
    Decorator to profile memory usage of functions that create tensors.
    
    Usage:
        @profile_tensor_memory("my_function")
        def my_function():
            return torch.randn(1000, 1000)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = MemoryProfiler()
            profiler.start(name)
            result = func(*args, **kwargs)
            profiler.end(name)
            profiler.print_report()
            return result
        return wrapper
    return decorator


# Additional utility functions for memory debugging

def print_gpu_memory_summary():
    """Print detailed GPU memory summary."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "="*80)
    print("GPU MEMORY SUMMARY")
    print("="*80)
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Allocated:     {torch.cuda.memory_allocated(i) / 1024**2:>10.1f} MB")
        print(f"  Reserved:      {torch.cuda.memory_reserved(i) / 1024**2:>10.1f} MB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:>10.1f} MB")
        print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(i) / 1024**2:>10.1f} MB")
        
        # Show percentage of total GPU memory
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**2
        allocated_pct = (torch.cuda.memory_allocated(i) / props.total_memory) * 100
        print(f"  Total Memory:  {total_memory:>10.1f} MB")
        print(f"  Usage:         {allocated_pct:>10.1f}%")
    print("="*80 + "\n")


def find_large_tensors(threshold_mb=100):
    """Find all tensors in memory larger than threshold."""
    import gc
    
    large_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                if size_mb > threshold_mb:
                    large_tensors.append({
                        'size_mb': size_mb,
                        'shape': tuple(obj.shape),
                        'dtype': obj.dtype,
                        'device': obj.device,
                        'requires_grad': obj.requires_grad
                    })
        except:
            pass
    
    large_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
    
    print(f"\nFound {len(large_tensors)} tensors > {threshold_mb} MB:")
    print("-"*80)
    for i, tensor_info in enumerate(large_tensors[:20]):  # Show top 20
        print(f"{i+1}. {tensor_info['size_mb']:>8.1f} MB - "
              f"shape: {str(tensor_info['shape']):<30} "
              f"dtype: {str(tensor_info['dtype']):<15} "
              f"device: {tensor_info['device']}")
    print("-"*80 + "\n")
    
    return large_tensors


def monitor_memory_continuously(duration_sec=60, interval_sec=1):
    """
    Monitor memory usage continuously and plot the results.
    Useful for finding memory leaks.
    """
    import time
    
    cpu_history = []
    gpu_history = []
    timestamps = []
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    print(f"Monitoring memory for {duration_sec} seconds...")
    
    while time.time() - start_time < duration_sec:
        timestamps.append(time.time() - start_time)
        cpu_history.append(process.memory_info().rss / 1024**2)
        
        if torch.cuda.is_available():
            gpu_history.append(torch.cuda.memory_allocated() / 1024**2)
        else:
            gpu_history.append(0)
        
        time.sleep(interval_sec)
    
    # Print summary
    print(f"\nMemory Monitoring Summary:")
    print(f"  CPU: {min(cpu_history):.1f} MB (min) → {max(cpu_history):.1f} MB (max)")
    if torch.cuda.is_available():
        print(f"  GPU: {min(gpu_history):.1f} MB (min) → {max(gpu_history):.1f} MB (max)")
    
    return timestamps, cpu_history, gpu_history