"""
Redis-based caching setup. 
Note: falls back to in-memory cache if Redis is unavailable.
"""
import os
import pickle
import hashlib
import functools
import threading
import logging
from typing import Any, Optional, Tuple
import pandas as pd
import redis
from redis.connection import ConnectionPool

logger = logging.getLogger(__name__)

try:
    REDIS_AVAILABLE = True

except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, falling back to in-memory cache")


class RedisCache:
    """
    Redis-based cache with fallback to in-memory cache.

    Features:
    - Connection pooling for performance
    - Automatic serialization/deserialization
    - TTL support for cache expiration
    - Fallback to in-memory cache if Redis unavailable
    - Thread-safe operations
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        default_ttl: int = 3600,
        max_memory_cache_size: int = 100,
        key_prefix: str = "isoformgazer:"
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host (default from REDIS_HOST env var or 'localhost')
            port: Redis port (default from REDIS_PORT env var or 6379)
            db: Redis database number
            password: Redis password (from REDIS_PASSWORD env var)
            default_ttl: Default TTL in seconds (1 hour default)
            max_memory_cache_size: Max size for fallback in-memory cache
            key_prefix: Prefix for all Redis keys
        """
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0
        }

        self.redis_client = None
        self.use_redis = False

        if REDIS_AVAILABLE:
            try:
                redis_host = host or os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(port or os.getenv('REDIS_PORT', 6379))
                redis_password = password or os.getenv('REDIS_PASSWORD')

                pool = ConnectionPool(
                    host=redis_host,
                    port=redis_port,
                    db=db,
                    password=redis_password if redis_password else None,
                    max_connections=20,
                    decode_responses=False,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )

                self.redis_client = redis.Redis(connection_pool=pool)
                self.redis_client.ping()
                self.use_redis = True
                logger.info(f"✓ Connected to Redis at {redis_host}:{redis_port}")

            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
                self.redis_client = None
                self.use_redis = False

        if not self.use_redis:
            from src.isoformgazer.performance_utils import SimpleCache
            self.memory_cache = SimpleCache(max_size=max_memory_cache_size)
            logger.info("Using in-memory cache as fallback")

    def _make_key(self, func, args, kwargs) -> str:
        """Generate cache key from function and arguments"""
        func_name = f"{func.__module__}.{func.__name__}"
        key_parts = [func_name]

        for arg in args:
            if isinstance(arg, (list, dict, set)):
                key_parts.append(hashlib.md5(str(sorted(str(arg))).encode()).hexdigest()[:16])
            elif isinstance(arg, pd.DataFrame):
                key_parts.append(hashlib.md5(str(arg.shape).encode()).hexdigest()[:16])
            else:
                key_parts.append(str(arg))

        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (list, dict, set)):
                key_parts.append(f"{k}={hashlib.md5(str(sorted(str(v))).encode()).hexdigest()[:16]}")
            elif isinstance(v, pd.DataFrame):
                key_parts.append(f"{k}={hashlib.md5(str(v.shape).encode()).hexdigest()[:16]}")
            else:
                key_parts.append(f"{k}={v}")

        key = ":".join(key_parts)
        return f"{self.key_prefix}{key}"

    def get(self, func, args, kwargs) -> Tuple[Any, bool]:
        """
        Get cached result for function call.

        Returns:
            Tuple of (result, found) where found is True if cache hit
        """
        try:
            key = self._make_key(func, args, kwargs)

            if self.use_redis:
                try:
                    cached_bytes = self.redis_client.get(key)
                    if cached_bytes is not None:
                        result = pickle.loads(cached_bytes)
                        with self._lock:
                            self.stats['hits'] += 1
                        return result, True
                    else:
                        with self._lock:
                            self.stats['misses'] += 1
                        return None, False

                except Exception as e:
                    logger.error(f"Redis get error: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

            # Fallback to memory cache
            if hasattr(self, 'memory_cache'):
                return self.memory_cache.get(func, args, kwargs)

            return None, False

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None, False

    def put(self, func, args, kwargs, result, ttl: int = None):
        """
        Store result in cache.

        Args:
            func: Function
            args: Function arguments
            kwargs: Function keyword arguments
            result: Result to cache
            ttl: Time-to-live in seconds (None = use default)
        """
        try:
            key = self._make_key(func, args, kwargs)
            ttl = ttl or self.default_ttl

            if self.use_redis:
                try:
                    cached_bytes = pickle.dumps(result)
                    self.redis_client.setex(key, ttl, cached_bytes)
                    return

                except Exception as e:
                    logger.error(f"Redis put error: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

            if hasattr(self, 'memory_cache'):
                self.memory_cache.put(func, args, kwargs, result)

        except Exception as e:
            logger.error(f"Cache put error: {e}")

    def get_value(self, key: str) -> Optional[Any]:
        """
        Get a value directly by key (for non-function caching).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            full_key = f"{self.key_prefix}{key}"

            if self.use_redis:
                try:
                    cached_bytes = self.redis_client.get(full_key)
                    if cached_bytes is not None:
                        with self._lock:
                            self.stats['hits'] += 1
                        return pickle.loads(cached_bytes)
                except Exception as e:
                    logger.error(f"Redis get_value error: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

            with self._lock:
                self.stats['misses'] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get_value error: {e}")
            return None

    def set_value(self, key: str, value: Any, ttl: int = None):
        """
        Set a value directly by key (for non-function caching).

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
        """
        try:
            full_key = f"{self.key_prefix}{key}"
            ttl = ttl or self.default_ttl

            if self.use_redis:
                try:
                    cached_bytes = pickle.dumps(value)
                    self.redis_client.setex(full_key, ttl, cached_bytes)
                except Exception as e:
                    logger.error(f"Redis set_value error: {e}")
                    with self._lock:
                        self.stats['errors'] += 1

        except Exception as e:
            logger.error(f"Cache set_value error: {e}")

    def delete(self, key: str):
        """Delete a key from cache"""
        try:
            full_key = f"{self.key_prefix}{key}"

            if self.use_redis:
                try:
                    self.redis_client.delete(full_key)
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")

        except Exception as e:
            logger.error(f"Cache delete error: {e}")

    def clear(self, pattern: str = None):
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match keys (e.g., "gene:*")
                    If None, clears all keys with this cache's prefix
        """
        try:
            if self.use_redis:
                try:
                    search_pattern = f"{self.key_prefix}{pattern}" if pattern else f"{self.key_prefix}*"
                    cursor = 0
                    while True:
                        cursor, keys = self.redis_client.scan(
                            cursor=cursor,
                            match=search_pattern,
                            count=100
                        )

                        if keys:
                            self.redis_client.delete(*keys)

                        if cursor == 0:
                            break

                    logger.info(f"Cleared cache with pattern: {search_pattern}")

                except Exception as e:
                    logger.error(f"Redis clear error: {e}")

            if hasattr(self, 'memory_cache'):
                self.memory_cache.cache.clear()
                self.memory_cache.access_times.clear()

        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self._lock:
            stats = self.stats.copy()

        total_requests = stats['hits'] + stats['misses']
        hit_rate = (stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        result = {
            'backend': 'redis' if self.use_redis else 'memory',
            'hits': stats['hits'],
            'misses': stats['misses'],
            'errors': stats['errors'],
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests
        }

        if self.use_redis and self.redis_client:
            try:
                info = self.redis_client.info('stats')
                result['redis_keys'] = self.redis_client.dbsize()
                result['redis_memory'] = self.redis_client.info('memory').get('used_memory_human', 'unknown')
            except:
                pass

        return result

    def health_check(self) -> dict:
        """Check cache health"""
        result = {
            'redis_available': REDIS_AVAILABLE,
            'redis_connected': self.use_redis,
            'backend': 'redis' if self.use_redis else 'memory'
        }

        if self.use_redis:
            try:
                self.redis_client.ping()
                result['redis_ping'] = 'ok'
            except Exception as e:
                result['redis_ping'] = f'failed: {e}'

        return result

cache = None

def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global cache
    if cache is None:
        cache = RedisCache()
    return cache


def cached(cache_timeout=3600):
    """
    Decorator to cache function results using Redis.

    Args:
        cache_timeout: TTL in seconds (default 1 hour)

    Example:
        @cached(cache_timeout=300)
        def expensive_function(arg1, arg2):
            return compute_result(arg1, arg2)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            c = get_cache()
            result, found = c.get(func, args, kwargs)
            if found:
                return result

            result = func(*args, **kwargs)
            c.put(func, args, kwargs, result, ttl=cache_timeout)
            return result

        return wrapper
    return decorator

CACHE_STATS = {'hits': 0, 'misses': 0}

def update_compat_stats():
    """Update compatibility stats from Redis cache"""
    c = get_cache()
    stats = c.get_stats()
    CACHE_STATS['hits'] = stats['hits']
    CACHE_STATS['misses'] = stats['misses']
