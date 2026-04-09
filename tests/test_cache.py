"""
Tests for bound propagation caching.
"""

import pytest
import torch

from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.cache import BoundCache, hash_region, regions_equal, region_contains
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods import (
    ForwardLBPPropagator,
    IBPPropagator,
)
from bound_propagation.regions.hyperrectangle import HyperRectangle
from bound_propagation.regions.multi_input import MultiInputRegion

# Import relaxations to ensure they're registered
import bound_propagation.relaxations  # noqa: F401


class TestRegionHashing:
    """Test region hashing utilities."""
    
    def test_hash_hyperrectangle(self):
        """Test hashing of HyperRectangle."""
        region1 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region2 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region3 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        
        # Same regions should have same hash
        assert hash_region(region1) == hash_region(region2)
        
        # Different regions should have different hash (with high probability)
        assert hash_region(region1) != hash_region(region3)
    
    def test_hash_multi_input_region(self):
        """Test hashing of MultiInputRegion."""
        region1 = MultiInputRegion({
            0: HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0])),
            1: HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
        })
        region2 = MultiInputRegion({
            0: HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0])),
            1: HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
        })
        region3 = MultiInputRegion({
            0: HyperRectangle(torch.tensor([0.0]), torch.tensor([2.0])),
            1: HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])),
        })
        
        # Same regions should have same hash
        assert hash_region(region1) == hash_region(region2)
        
        # Different regions should have different hash
        assert hash_region(region1) != hash_region(region3)
    
    def test_regions_equal(self):
        """Test region equality checking."""
        region1 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region2 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        region3 = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        
        assert regions_equal(region1, region2)
        assert not regions_equal(region1, region3)
    
    def test_region_contains(self):
        """Test region containment checking."""
        outer = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        inner = HyperRectangle(
            lower=torch.tensor([0.5, 0.5]),
            upper=torch.tensor([1.5, 1.5]),
        )
        disjoint = HyperRectangle(
            lower=torch.tensor([3.0, 3.0]),
            upper=torch.tensor([4.0, 4.0]),
        )
        
        assert region_contains(outer, inner)
        assert region_contains(outer, outer)  # Contains itself
        assert not region_contains(inner, outer)
        assert not region_contains(outer, disjoint)


class TestBoundCache:
    """Test BoundCache class."""
    
    def test_create_cache(self):
        """Test creating a cache."""
        cache = BoundCache()
        assert len(cache) == 0
        assert cache.get_stats()["cache_size"] == 0
    
    def test_store_and_retrieve(self):
        """Test storing and retrieving bounds."""
        cache = BoundCache()
        
        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = IntervalBounds(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        
        # Store bounds
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        assert len(cache) == 1
        
        # Retrieve bounds
        retrieved = cache.get(node_id=5, method="ibp", region=region)
        assert retrieved is not None
        assert torch.allclose(retrieved.lower, bounds.lower)
        assert torch.allclose(retrieved.upper, bounds.upper)
    
    def test_cache_miss(self):
        """Test cache miss scenarios."""
        cache = BoundCache()
        
        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        
        # Different node_id
        assert cache.get(node_id=5, method="ibp", region=region) is None
        
        # Store something
        bounds = IntervalBounds(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([1.0, 1.0]))
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        
        # Different node_id -> miss
        assert cache.get(node_id=6, method="ibp", region=region) is None
        
        # Different method -> miss
        assert cache.get(node_id=5, method="forward_lbp", region=region) is None
        
        # Different region -> miss
        different_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        assert cache.get(node_id=5, method="ibp", region=different_region) is None
    
    def test_conservative_reuse(self):
        """Test conservative reuse of cached bounds."""
        cache = BoundCache(enable_conservative_reuse=True)
        
        # Cache bounds for a large region
        large_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        bounds = IntervalBounds(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        cache.store(node_id=5, method="ibp", region=large_region, bounds=bounds)
        
        # Query with a smaller contained region
        small_region = HyperRectangle(
            lower=torch.tensor([0.5, 0.5]),
            upper=torch.tensor([1.5, 1.5]),
        )
        retrieved = cache.get(node_id=5, method="ibp", region=small_region)
        
        # Should get bounds from larger region
        assert retrieved is not None
        assert torch.allclose(retrieved.lower, bounds.lower)
        
        # Check stats
        stats = cache.get_stats()
        assert stats["conservative_hits"] == 1
    
    def test_conservative_reuse_disabled(self):
        """Test that conservative reuse is disabled by default."""
        cache = BoundCache(enable_conservative_reuse=False)
        
        # Cache bounds for a large region
        large_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        bounds = IntervalBounds(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        cache.store(node_id=5, method="ibp", region=large_region, bounds=bounds)
        
        # Query with a smaller contained region
        small_region = HyperRectangle(
            lower=torch.tensor([0.5, 0.5]),
            upper=torch.tensor([1.5, 1.5]),
        )
        retrieved = cache.get(node_id=5, method="ibp", region=small_region)
        
        # Should get None (conservative reuse disabled)
        assert retrieved is None
    
    def test_invalidate_node(self):
        """Test invalidating a specific node."""
        cache = BoundCache()
        
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds1 = IntervalBounds(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
        bounds2 = IntervalBounds(lower=torch.tensor([0.0]), upper=torch.tensor([2.0]))
        
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds1)
        cache.store(node_id=6, method="ibp", region=region, bounds=bounds2)
        assert len(cache) == 2
        
        # Invalidate node 5
        cache.invalidate_node(5)
        assert len(cache) == 1
        assert cache.get(node_id=5, method="ibp", region=region) is None
        assert cache.get(node_id=6, method="ibp", region=region) is not None
    
    def test_invalidate_method(self):
        """Test invalidating a specific method."""
        cache = BoundCache()
        
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds = IntervalBounds(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
        
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        cache.store(node_id=5, method="forward_lbp", region=region, bounds=bounds)
        assert len(cache) == 2
        
        # Invalidate ibp method
        cache.invalidate_method("ibp")
        assert len(cache) == 1
        assert cache.get(node_id=5, method="ibp", region=region) is None
        assert cache.get(node_id=5, method="forward_lbp", region=region) is not None
    
    def test_clear_cache(self):
        """Test clearing the entire cache."""
        cache = BoundCache()
        
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds = IntervalBounds(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
        
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        cache.store(node_id=6, method="ibp", region=region, bounds=bounds)
        assert len(cache) == 2
        
        cache.clear()
        assert len(cache) == 0
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = BoundCache()
        
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds = IntervalBounds(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["stores"] == 0
        
        # Store
        cache.store(node_id=5, method="ibp", region=region, bounds=bounds)
        assert cache.get_stats()["stores"] == 1
        
        # Hit
        cache.get(node_id=5, method="ibp", region=region)
        assert cache.get_stats()["hits"] == 1
        
        # Miss
        cache.get(node_id=6, method="ibp", region=region)
        assert cache.get_stats()["misses"] == 1
        
        # Hit rate
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5  # 1 hit out of 2 queries


class TestCachedPropagation:
    """Test propagators with caching enabled."""
    
    def test_ibp_with_cache(self):
        """Test IBP propagator with caching."""
        # Create simple graph: x -> relu -> output
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")
        
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        
        relu_node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(relu_node)
        graph.mark_outputs([relu_node])
        
        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        
        # Create cache and propagator
        cache = BoundCache()
        propagator = IBPPropagator(cache=cache)
        
        # First propagation - should populate cache
        bounds1 = propagator.propagate(graph, region)
        stats1 = cache.get_stats()
        assert stats1["stores"] == 2  # Input + ReLU
        assert stats1["hits"] == 0
        
        # Second propagation - should use cache
        bounds2 = propagator.propagate(graph, region)
        stats2 = cache.get_stats()
        assert stats2["hits"] == 2  # Both nodes from cache
        
        # Results should be identical
        assert torch.allclose(bounds1[1].lower, bounds2[1].lower)
        assert torch.allclose(bounds1[1].upper, bounds2[1].upper)
    
    def test_forward_lbp_with_cache(self):
        """Test Forward LBP with caching."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")
        
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        
        sigmoid_node = Node(
            id=1,
            op_type=OperationType.SIGMOID,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(sigmoid_node)
        graph.mark_outputs([sigmoid_node])
        
        region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        
        cache = BoundCache()
        propagator = ForwardLBPPropagator(cache=cache)
        
        # First propagation
        bounds1 = propagator.propagate(graph, region)
        assert cache.get_stats()["stores"] == 2
        
        # Second propagation
        bounds2 = propagator.propagate(graph, region)
        assert cache.get_stats()["hits"] == 2
        
        # Bounds should be identical
        assert torch.allclose(bounds1[1].lower, bounds2[1].lower)
        assert torch.allclose(bounds1[1].upper, bounds2[1].upper)
    
    def test_cache_with_different_regions(self):
        """Test caching with different input regions."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")
        
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        graph.mark_outputs([input_node])
        
        region1 = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        region2 = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0]))
        
        cache = BoundCache()
        propagator = IBPPropagator(cache=cache)
        
        # Propagate with region1
        bounds1 = propagator.propagate(graph, region1)
        assert cache.get_stats()["stores"] == 1
        
        # Propagate with region2 (different region)
        bounds2 = propagator.propagate(graph, region2)
        assert cache.get_stats()["stores"] == 2  # New entry
        assert cache.get_stats()["hits"] == 0  # No hits (different region)
        
        # Propagate with region1 again
        bounds3 = propagator.propagate(graph, region1)
        assert cache.get_stats()["hits"] == 1  # Hit from first call
