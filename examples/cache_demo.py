"""
Performance demonstration for bound caching.

Shows the performance improvement from using caching.
"""

import time
import torch

from bound_propagation.cache import BoundCache
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods import IBPPropagator
from bound_propagation.regions.hyperrectangle import HyperRectangle

# Import relaxations
import bound_propagation.relaxations  # noqa: F401


def create_test_network():
    """Create a simple test network: x -> relu -> sigmoid -> tanh."""
    graph = Graph()
    metadata = TensorMetadata(shape=(100,), dtype="float32")
    
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
    
    sigmoid_node = Node(
        id=2,
        op_type=OperationType.SIGMOID,
        inputs=[relu_node],
        output_metadata=metadata,
        attributes={},
        node_type=NodeType.OPERATION,
    )
    graph.add_node(sigmoid_node)
    
    tanh_node = Node(
        id=3,
        op_type=OperationType.TANH,
        inputs=[sigmoid_node],
        output_metadata=metadata,
        attributes={},
        node_type=NodeType.OPERATION,
    )
    graph.add_node(tanh_node)
    
    graph.mark_outputs([tanh_node])
    return graph


def benchmark_without_cache(graph, region, num_runs=100):
    """Benchmark propagation without caching."""
    propagator = IBPPropagator(cache=None)
    
    start = time.time()
    for _ in range(num_runs):
        bounds = propagator.propagate(graph, region)
    end = time.time()
    
    return end - start, bounds


def benchmark_with_cache(graph, region, num_runs=100):
    """Benchmark propagation with caching."""
    cache = BoundCache()
    propagator = IBPPropagator(cache=cache)
    
    start = time.time()
    for _ in range(num_runs):
        bounds = propagator.propagate(graph, region)
    end = time.time()
    
    stats = cache.get_stats()
    return end - start, bounds, stats


if __name__ == "__main__":
    print("=" * 70)
    print("Bound Propagation Caching Performance Demonstration")
    print("=" * 70)
    
    # Create test network
    graph = create_test_network()
    region = HyperRectangle(
        lower=torch.full((100,), -1.0),
        upper=torch.full((100,), 1.0),
    )
    
    num_runs = 100
    print(f"\nRunning {num_runs} propagations on network: input -> relu -> sigmoid -> tanh")
    print(f"Tensor size: {region.shape}")
    
    # Benchmark without cache
    print("\n" + "-" * 70)
    print("WITHOUT CACHING:")
    print("-" * 70)
    time_no_cache, bounds_no_cache = benchmark_without_cache(graph, region, num_runs)
    print(f"Total time: {time_no_cache:.4f} seconds")
    print(f"Average per run: {time_no_cache/num_runs*1000:.2f} ms")
    
    # Benchmark with cache
    print("\n" + "-" * 70)
    print("WITH CACHING:")
    print("-" * 70)
    time_with_cache, bounds_with_cache, stats = benchmark_with_cache(graph, region, num_runs)
    print(f"Total time: {time_with_cache:.4f} seconds")
    print(f"Average per run: {time_with_cache/num_runs*1000:.2f} ms")
    print(f"\nCache statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Stores: {stats['stores']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"  Cache size: {stats['cache_size']} entries")
    
    # Speedup
    print("\n" + "=" * 70)
    speedup = time_no_cache / time_with_cache
    print(f"SPEEDUP: {speedup:.2f}x faster with caching")
    print(f"Time saved: {(time_no_cache - time_with_cache):.4f} seconds")
    print("=" * 70)
    
    # Verify results are identical
    assert torch.allclose(bounds_no_cache[3].lower, bounds_with_cache[3].lower)
    assert torch.allclose(bounds_no_cache[3].upper, bounds_with_cache[3].upper)
    print("\n✓ Results verified: cached and non-cached bounds are identical")
