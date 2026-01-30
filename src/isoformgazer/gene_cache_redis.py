"""
Redis caching setup for gene query dropdown. 
"""
import logging
from typing import List, Dict, Optional
from src.isoformgazer.redis_cache import get_cache

logger = logging.getLogger(__name__)
DEFAULT_GENE_CACHE_KEY = "gene_cache:default"

GENE_LIST_TTL = 604800
DEFAULT_GENE_TTL = 604800


def _get_gene_list_key(species: str = "Human") -> str:
    """
    Generate species-specific cache key for gene list.

    Args:
        species: Species name ("Human" or "Mouse")

    Returns:
        Cache key string
    """
    return f"gene_list:{species.lower()}"


def cache_gene_list(gene_options: List[Dict[str, str]], species: str = "Human"):
    """
    Cache the full gene list for dropdown.

    Args:
        gene_options: List of dicts with 'label' and 'value' keys
        species: Species name ("Human" or "Mouse")

    Example:
        gene_options = [
            {'label': 'A1BG', 'value': 'A1BG'},
            {'label': 'A1BG-AS1', 'value': 'A1BG-AS1'},
            ...
        ]
    """
    try:
        cache = get_cache()
        cache_key = _get_gene_list_key(species)
        cache.set_value(cache_key, gene_options, ttl=GENE_LIST_TTL)
        logger.info(f"Cached {len(gene_options)} {species} genes to Redis (key: {cache_key})")
    except Exception as e:
        logger.error(f"Failed to cache gene list for {species}: {e}")


def get_cached_gene_list(species: str = "Human") -> Optional[List[Dict[str, str]]]:
    """
    Get cached gene list for dropdown.

    Args:
        species: Species name ("Human" or "Mouse")

    Returns:
        List of gene options or None if not cached
    """
    try:
        cache = get_cache()
        cache_key = _get_gene_list_key(species)
        gene_options = cache.get_value(cache_key)

        if gene_options:
            logger.debug(f"Retrieved {len(gene_options)} {species} genes from Redis cache (key: {cache_key})")
            return gene_options
        else:
            logger.debug(f"No gene list found in cache for {species} (key: {cache_key})")
            return None

    except Exception as e:
        logger.error(f"Failed to get cached gene list for {species}: {e}")
        return None


def cache_default_gene_data(cache_data: Dict):
    """
    Cache default gene data (plots and metadata).

    Args:
        cache_data: Dictionary containing:
            - plots: Plotly figure objects
            - metadata: Gene metadata
            - timestamp: Cache creation time

    Example:
        cache_data = {
            'plots': {
                'structure': fig1,
                'clustergram': fig2,
                ...
            },
            'metadata': {...},
            'timestamp': datetime.now().isoformat()
        }
    """
    try:
        cache = get_cache()
        cache.set_value(DEFAULT_GENE_CACHE_KEY, cache_data, ttl=DEFAULT_GENE_TTL)
        logger.info("Cached default gene data to Redis")
    except Exception as e:
        logger.error(f"Failed to cache default gene data: {e}")


def get_cached_default_gene_data() -> Optional[Dict]:
    """
    Get cached default gene data.

    Returns:
        Dictionary with cached data or None if not cached
    """
    try:
        cache = get_cache()
        cache_data = cache.get_value(DEFAULT_GENE_CACHE_KEY)

        if cache_data:
            logger.debug("Retrieved default gene data from Redis cache")
            return cache_data
        else:
            logger.debug("No default gene data found in cache")
            return None

    except Exception as e:
        logger.error(f"Failed to get cached default gene data: {e}")
        return None


def invalidate_gene_list_cache(species: str = None):
    """
    Invalidate gene list cache (e.g., after database update).

    Args:
        species: Species to invalidate ("Human", "Mouse", or None for all species)
    """
    try:
        cache = get_cache()
        if species is None:
            # Invalidate both Human and Mouse caches
            for sp in ["Human", "Mouse"]:
                cache_key = _get_gene_list_key(sp)
                cache.delete(cache_key)
            logger.info("Invalidated gene list cache for all species")
        else:
            cache_key = _get_gene_list_key(species)
            cache.delete(cache_key)
            logger.info(f"Invalidated gene list cache for {species}")
    except Exception as e:
        logger.error(f"Failed to invalidate gene list cache: {e}")


def invalidate_default_gene_cache():
    """Invalidate default gene cache"""
    try:
        cache = get_cache()
        cache.delete(DEFAULT_GENE_CACHE_KEY)
        logger.info("Invalidated default gene cache")
    except Exception as e:
        logger.error(f"Failed to invalidate default gene cache: {e}")


def clear_all_gene_caches():
    """Clear all gene-related caches"""
    try:
        cache = get_cache()
        cache.clear(pattern="gene*")
        logger.info("Cleared all gene caches")
    except Exception as e:
        logger.error(f"Failed to clear gene caches: {e}")


def get_cache_stats() -> Dict:
    """Get statistics about gene caching"""
    try:
        cache = get_cache()
        stats = cache.get_stats()

        # Check both species-specific gene lists
        human_genes_cached = get_cached_gene_list("Human") is not None
        mouse_genes_cached = get_cached_gene_list("Mouse") is not None
        default_gene_cached = get_cached_default_gene_data() is not None

        stats['gene_list_cached'] = human_genes_cached or mouse_genes_cached
        stats['human_genes_cached'] = human_genes_cached
        stats['mouse_genes_cached'] = mouse_genes_cached
        stats['default_gene_cached'] = default_gene_cached

        return stats

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {
            'error': str(e),
            'gene_list_cached': False,
            'human_genes_cached': False,
            'mouse_genes_cached': False,
            'default_gene_cached': False
        }
