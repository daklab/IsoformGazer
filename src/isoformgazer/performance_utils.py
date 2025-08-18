"""
Performance profiling and optimization utilities for IsoformGazer
"""
import time
import functools
import importlib
import sqlite3
import pandas as pd
import numpy as np
import cProfile
import pstats
import io
import psutil
import os
from typing import Dict, Any, Callable, Optional
from memory_profiler import profile as memory_profile
import logging
from collections import defaultdict
import hashlib
import pickle
import threading
from pathlib import Path

############################################################################################
# Disable logging for production use, currently set to CRITICAL to silence all output
############################################################################################
logging.basicConfig(level=logging.CRITICAL)
performance_logger = logging.getLogger('performance')
performance_logger.setLevel(logging.CRITICAL)

############################################################################################
# GLOBAL PERFORMANCE STATS TRACKING
############################################################################################
PERFORMANCE_STATS = defaultdict(list)
QUERY_STATS = defaultdict(list)
CACHE_STATS = {'hits': 0, 'misses': 0, 'size': 0}

class PerformanceProfiler:
    """Comprehensive performance profiler to help make optimizations easier to find"""
    def __init__(self, enable_memory_profiling=True):
        self.enable_memory_profiling = enable_memory_profiling
        self.profiler = None
        self.memory_usage = []
        self.query_times = {}
        
    def start_profiling(self):
        """Start CPU profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
    def stop_profiling(self, output_file="performance_profile.txt"):
        """Stop profiling and save results"""
        if self.profiler:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(50)
            
            with open(output_file, 'w') as f:
                f.write(s.getvalue())
            
            performance_logger.info(f"Performance profile saved to {output_file}")
            return s.getvalue()

profiler = PerformanceProfiler()

def timing_decorator(log_to_file=True, threshold_seconds=0.1):
    """Decorator to measure function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Stores performance stats
                PERFORMANCE_STATS[func.__name__].append({
                    'time': execution_time,
                    'memory_delta': memory_delta,
                    'success': success,
                    'error': error,
                    'timestamp': time.time()
                })

                if execution_time > threshold_seconds:
                    performance_logger.warning(
                        f"{func.__name__} took {execution_time:.3f}s, "
                        f"memory delta: {memory_delta:.1f}MB"
                    )
                
                if log_to_file and execution_time > threshold_seconds:
                    log_performance_event(func.__name__, execution_time, memory_delta)
            
            return result
        
        return wrapper
    return decorator


def query_profiler(func):
    """Decorator specifically for database query profiling"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Try to extract query info from args/kwargs
        query_info = "unknown"
        if args and isinstance(args[0], str):
            query_info = args[0][:100] + "..." if len(args[0]) > 100 else args[0]
        
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        
        QUERY_STATS[func.__name__].append({
            'time': execution_time,
            'query': query_info,
            'timestamp': time.time()
        })
        # logging for identifying super slow queries
        if execution_time > 0.5:
            performance_logger.warning(f"Slow query in {func.__name__}: {execution_time:.3f}s")
        
        return result
    return wrapper


def log_performance_event(function_name, execution_time, memory_delta):
    """Log performance events to file"""
    log_dir = Path("performance_logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"performance_{time.strftime('%Y%m%d')}.log"
    
    with open(log_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{function_name},{execution_time:.3f},{memory_delta:.1f}\n")


def get_performance_summary():
    """Get summary of performance statistics"""
    summary = {}
    for func_name, stats in PERFORMANCE_STATS.items():
        times = [s['time'] for s in stats if s['success']]
        if times:
            summary[func_name] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'min_time': np.min(times),
                'total_time': np.sum(times),
                'errors': len([s for s in stats if not s['success']])
            }
    
    return summary


def get_query_summary():
    """Get summary of query performance"""
    summary = {}
    
    for func_name, stats in QUERY_STATS.items():
        times = [s['time'] for s in stats]
        if times:
            summary[func_name] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'total_time': np.sum(times)
            }
    
    return summary


class MemoryTracker:
    """Track memory usage during operations"""
    def __init__(self):
        self.baseline = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak = self.baseline
        self.measurements = []
    
    def measure(self, label=""):
        """Take a memory measurement"""
        current = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak = max(self.peak, current)
        self.measurements.append({
            'label': label,
            'memory_mb': current,
            'delta_from_baseline': current - self.baseline,
            'timestamp': time.time()
        })
        return current
    
    def get_summary(self):
        """Get memory usage summary"""
        return {
            'baseline_mb': self.baseline,
            'peak_mb': self.peak,
            'peak_delta_mb': self.peak - self.baseline,
            'measurements': len(self.measurements)
        }

memory_tracker = MemoryTracker()

class SimpleCache:
    """Simple in-memory cache with size limits"""
    def __init__(self, max_size=100, max_memory_mb=500):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._lock = threading.Lock()
    
    def _get_cache_key(self, func, args, kwargs):
        """Generate cache key from function and arguments"""
        # Create a hashable key from function name and arguments
        key_data = (func.__name__, str(args), str(sorted(kwargs.items())))
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _estimate_size_mb(self, obj):
        """Estimate object size in MB"""
        try:
            return len(pickle.dumps(obj)) / 1024 / 1024
        except:
            return 0
    
    def _evict_if_needed(self):
        """Evict old entries if cache is too large"""
        # Evict by access time (LRU)
        while len(self.cache) > self.max_size:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Also evict by memory usage
        total_memory = sum(self._estimate_size_mb(obj) for obj in self.cache.values())
        while total_memory > self.max_memory_mb and self.cache:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            obj_size = self._estimate_size_mb(self.cache[oldest_key])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            total_memory -= obj_size
    
    def get(self, func, args, kwargs):
        """Get cached result"""
        key = self._get_cache_key(func, args, kwargs)
        
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                CACHE_STATS['hits'] += 1
                return self.cache[key], True
            else:
                CACHE_STATS['misses'] += 1
                return None, False
    
    def put(self, func, args, kwargs, result):
        """Store result in cache"""
        key = self._get_cache_key(func, args, kwargs)
        
        with self._lock:
            self.cache[key] = result
            self.access_times[key] = time.time()
            CACHE_STATS['size'] = len(self.cache)
            self._evict_if_needed()
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            CACHE_STATS['size'] = 0
    
    def get_stats(self):
        """Get cache statistics"""
        total_requests = CACHE_STATS['hits'] + CACHE_STATS['misses']
        hit_rate = CACHE_STATS['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': CACHE_STATS['hits'],
            'misses': CACHE_STATS['misses'],
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'memory_mb': sum(self._estimate_size_mb(obj) for obj in self.cache.values())
        }

cache = SimpleCache()

def cached(cache_timeout=300):
    """Decorator to cache function results"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result, found = cache.get(func, args, kwargs)
            if found:
                return result
            
            result = func(*args, **kwargs)
            cache.put(func, args, kwargs, result)
            
            return result
        return wrapper
    return decorator


def profile_database_operations():
    """Profile database operations specifically"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            db_path = None
            if args and isinstance(args[0], str) and args[0].endswith('.db'):
                db_path = args[0]
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            performance_logger.info(
                f"DB operation {func.__name__}: {execution_time:.3f}s"
            )
            
            return result
        return wrapper
    return decorator


def monitor_memory_usage(func):
    """Decorator to monitor memory usage during function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracker = MemoryTracker()
        tracker.measure("start")
        
        result = func(*args, **kwargs)
        
        tracker.measure("end")
        summary = tracker.get_summary()
        
        # note: only logs if peak memory usage exceeds 100MB
        if summary['peak_delta_mb'] > 100:
            performance_logger.warning(
                f"{func.__name__} used {summary['peak_delta_mb']:.1f}MB memory"
            )
        
        return result
    return wrapper


def create_performance_report():
    """Create comprehensive performance report"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'function_performance': get_performance_summary(),
        'query_performance': get_query_summary(),
        'cache_stats': cache.get_stats(),
        'memory_stats': memory_tracker.get_summary()
    }
    report_file = f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    performance_logger.info(f"Performance report saved to {report_file}")
    return report


class ProfilerContext:
    """Context manager for profiling specific code blocks"""
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        performance_logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        performance_logger.info(
            f"Completed {self.operation_name}: {execution_time:.3f}s, "
            f"memory delta: {memory_delta:.1f}MB"
        )
        
        if exc_type:
            performance_logger.error(f"Error in {self.operation_name}: {exc_val}")


class MemoryTracker:
    """Track memory usage during operations"""
    def __init__(self):
        self.baseline = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak = self.baseline
        self.measurements = []
    
    def measure(self, label=""):
        """Take a memory measurement"""
        current = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak = max(self.peak, current)
        self.measurements.append({
            'label': label,
            'memory_mb': current,
            'delta_from_baseline': current - self.baseline,
            'timestamp': time.time()
        })
        return current
    
    def get_summary(self):
        """Get memory usage summary"""
        return {
            'baseline_mb': self.baseline,
            'peak_mb': self.peak,
            'peak_delta_mb': self.peak - self.baseline,
            'measurements': len(self.measurements)
        }

memory_tracker = MemoryTracker()

@timing_decorator(threshold_seconds=0.5)
def cached_transcript_structure_processing(db_path: str, gene_name: str, filtered_ids: list):
    """Cached version of transcript structure processing"""
    isoform_utils = importlib.import_module('isoform_utils')
    return isoform_utils.process_transcript_structure(db_path, gene_name, filtered_ids)


@timing_decorator(threshold_seconds=0.5)
def cached_atse_data_processing(gene_name: str, db_path: str, filtered_junction_ids=None):
    """Cached version of ATSE data processing"""
    junction_utils = importlib.import_module('junction_utils')
    return junction_utils.process_gene_atse_data(gene_name, db_path, filtered_junction_ids)


class PlotElementsOptimizer:
    """Optimizes plot element creation for better performance"""
    @staticmethod
    def batch_create_shapes(shapes_data: list) -> list:
        """Efficiently batch create plot shapes"""
        if not shapes_data:
            return []
        shapes = []
        shapes.extend(shapes_data)
        return shapes
    
    @staticmethod
    def preprocess_dataframe_for_plotting(df):
        """Optimize DataFrame for plotting operations"""
        if df.empty:
            return df
        
        df_opt = df.copy()
        numeric_cols = df_opt.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
            except (ValueError, TypeError):
                pass
            try:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
            except (ValueError, TypeError):
                pass
        
        return df_opt

plot_optimizer = PlotElementsOptimizer()


def precompute_plot_layout_data(transcript_data, junctions=None):
    """Precompute layout data for faster plot generation"""
    layout_data = {}

    if not transcript_data.empty:
        # Precompute transcript bounds
        layout_data['min_start'] = transcript_data['transcript_start'].min()
        layout_data['max_end'] = transcript_data['transcript_end'].max()
        layout_data['transcript_count'] = transcript_data['id'].nunique()
    
    if junctions:
        # Precompute junction bounds
        all_coords = []
        for j in junctions:
            all_coords.extend([j['start'], j['end']])
        
        layout_data['junction_min'] = min(all_coords) if all_coords else 0
        layout_data['junction_max'] = max(all_coords) if all_coords else 100000
        layout_data['junction_count'] = len(junctions)
    
    return layout_data


class PlotPerformanceContext:
    """Context manager for optimizing plot performance"""
    def __init__(self, gene_name: str, plot_type: str):
        self.gene_name = gene_name
        self.plot_type = plot_type
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        try:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except:
            self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        if execution_time > 5.0:
            performance_logger.info(f"Plot performance: {self.plot_type} for {self.gene_name} took {execution_time:.2f}s")