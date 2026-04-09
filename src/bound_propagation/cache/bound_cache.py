"""
Bound propagation caching system.

Caches computed bounds to avoid redundant computation. Supports:
- Exact cache hits (same node, method, and region)
- Conservative reuse (cached region contains query region)
- Cache invalidation when graph changes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from bound_propagation.cache.region_hash import hash_region, region_contains, regions_equal

if TYPE_CHECKING:
    from bound_propagation.bounds.abstract_bounds import AbstractBounds
    from bound_propagation.regions.abstract import AbstractRegion
    from bound_propagation.regions.hyperrectangle import HyperRectangle


@dataclass(frozen=True)
class CacheKey:
    """
    Key for bound cache lookups.
    
    Identifies a unique bound propagation computation by:
    - node_id: Which node's bounds we're computing
    - method: Which propagation method (ibp, forward_lbp, backward_lbp)
    - region_hash: Hash of the input region
    """
    
    node_id: int
    method: str
    region_hash: str
    
    def __hash__(self) -> int:
        """Hash for dictionary lookup."""
        return hash((self.node_id, self.method, self.region_hash))
    
    def __eq__(self, other) -> bool:
        """Equality check for dictionary lookup."""
        if not isinstance(other, CacheKey):
            return False
        return (
            self.node_id == other.node_id
            and self.method == other.method
            and self.region_hash == other.region_hash
        )


@dataclass
class CacheEntry:
    """
    Cached bound propagation result.
    
    Stores:
    - bounds: The computed bounds
    - region: The region these bounds were computed for
    - hit_count: Number of times this entry was retrieved
    """
    
    bounds: AbstractBounds
    region: AbstractRegion
    hit_count: int = 0


class BoundCache:
    """
    Cache for bound propagation results.
    
    Stores bounds computed for nodes and reuses them when possible.
    Supports exact matching and conservative reuse for contained regions.
    
    Example:
        >>> cache = BoundCache()
        >>> # After propagating bounds
        >>> cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        >>> # Later, retrieve cached bounds
        >>> cached = cache.get(node_id=5, method="ibp", region=region)
    """
    
    def __init__(self, *, enable_conservative_reuse: bool = False) -> None:
        """
        Initialize bound cache.
        
        Args:
            enable_conservative_reuse: If True, allow reusing bounds from
                larger regions when exact match not found. Useful for
                branch-and-bound, but may give looser bounds.
        """
        self._cache: Dict[CacheKey, CacheEntry] = {}
        self._enable_conservative_reuse = enable_conservative_reuse
        self._stats = {
            "hits": 0,
            "misses": 0,
            "conservative_hits": 0,
            "stores": 0,
        }
    
    def get(
        self,
        node_id: int,
        method: str,
        region: AbstractRegion,
    ) -> Optional[AbstractBounds]:
        """
        Retrieve cached bounds for a node.
        
        Args:
            node_id: Node identifier
            method: Propagation method name
            region: Input region
        
        Returns:
            Cached bounds if available, None otherwise
        """
        # Try exact match first
        region_hash = hash_region(region)
        key = CacheKey(node_id=node_id, method=method, region_hash=region_hash)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Verify region equality (hash collision check)
            if regions_equal(entry.region, region):
                entry.hit_count += 1
                self._stats["hits"] += 1
                return entry.bounds
        
        # Try conservative reuse if enabled
        if self._enable_conservative_reuse:
            cached_bounds = self._try_conservative_reuse(node_id, method, region)
            if cached_bounds is not None:
                self._stats["conservative_hits"] += 1
                return cached_bounds
        
        self._stats["misses"] += 1
        return None
    
    def _try_conservative_reuse(
        self,
        node_id: int,
        method: str,
        region: AbstractRegion,
    ) -> Optional[AbstractBounds]:
        """
        Try to reuse cached bounds from a larger region.
        
        If we have cached bounds for a region that contains the query region,
        those bounds are valid (though possibly loose) for the query.
        
        Args:
            node_id: Node identifier
            method: Propagation method name
            region: Input region
        
        Returns:
            Cached bounds if a containing region is found, None otherwise
        """
        from bound_propagation.regions.hyperrectangle import HyperRectangle
        
        # Conservative reuse only works for HyperRectangle currently
        if not isinstance(region, HyperRectangle):
            return None
        
        # Search for cached entries with same node_id and method
        for key, entry in self._cache.items():
            if key.node_id != node_id or key.method != method:
                continue
            
            # Check if cached region contains query region
            if isinstance(entry.region, HyperRectangle):
                if region_contains(entry.region, region):
                    entry.hit_count += 1
                    return entry.bounds
        
        return None
    
    def store(
        self,
        node_id: int,
        method: str,
        region: AbstractRegion,
        bounds: AbstractBounds,
    ) -> None:
        """
        Store bounds in the cache.
        
        Args:
            node_id: Node identifier
            method: Propagation method name
            region: Input region these bounds were computed for
            bounds: Computed bounds to cache
        """
        region_hash = hash_region(region)
        key = CacheKey(node_id=node_id, method=method, region_hash=region_hash)
        
        # Store with a copy of the region to avoid mutations
        self._cache[key] = CacheEntry(bounds=bounds, region=region)
        self._stats["stores"] += 1
    
    def invalidate_node(self, node_id: int) -> None:
        """
        Invalidate all cached entries for a specific node.
        
        Use when a node's operation or attributes change.
        
        Args:
            node_id: Node to invalidate
        """
        keys_to_remove = [key for key in self._cache if key.node_id == node_id]
        for key in keys_to_remove:
            del self._cache[key]
    
    def invalidate_method(self, method: str) -> None:
        """
        Invalidate all cached entries for a specific method.
        
        Args:
            method: Method name to invalidate
        """
        keys_to_remove = [key for key in self._cache if key.method == method]
        for key in keys_to_remove:
            del self._cache[key]
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        # Keep stats
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss/store counts
        """
        stats = self._stats.copy()
        stats["total_queries"] = stats["hits"] + stats["misses"]
        
        if stats["total_queries"] > 0:
            stats["hit_rate"] = stats["hits"] / stats["total_queries"]
            stats["effective_hit_rate"] = (
                stats["hits"] + stats["conservative_hits"]
            ) / stats["total_queries"]
        else:
            stats["hit_rate"] = 0.0
            stats["effective_hit_rate"] = 0.0
        
        stats["cache_size"] = len(self._cache)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "conservative_hits": 0,
            "stores": 0,
        }
    
    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: Tuple[int, str, str]) -> bool:
        """
        Check if cache contains an entry.
        
        Args:
            key: Tuple of (node_id, method, region_hash)
        
        Returns:
            True if cache contains this entry
        """
        cache_key = CacheKey(node_id=key[0], method=key[1], region_hash=key[2])
        return cache_key in self._cache
