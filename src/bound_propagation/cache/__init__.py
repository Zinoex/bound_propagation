"""
Caching system for bound propagation.

Provides caching of computed bounds to improve performance when
propagating through the same graph multiple times with different
or similar input regions.
"""

from bound_propagation.cache.bound_cache import BoundCache, CacheEntry, CacheKey
from bound_propagation.cache.region_hash import (
    hash_region,
    regions_equal,
    region_contains,
)

__all__ = [
    "BoundCache",
    "CacheKey",
    "CacheEntry",
    "hash_region",
    "regions_equal",
    "region_contains",
]
