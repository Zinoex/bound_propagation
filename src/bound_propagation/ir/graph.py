"""
Computation graph representation and utilities.

A Graph is a directed acyclic graph (DAG) of operations that can be analyzed
with bound propagation methods.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from .node import Node


class Graph:
    """
    Directed acyclic graph (DAG) representing a computation.

    A Graph manages a collection of nodes and provides:
    - Topological ordering for forward/backward passes
    - Input/output node tracking
    - Node lookup and iteration
    - Validation of graph structure

    Attributes:
        nodes: All nodes in the graph (ordered by ID)
        input_nodes: Nodes that are graph inputs
        output_nodes: Nodes that are graph outputs
    """

    def __init__(self, nodes: Sequence[Node] | None = None) -> None:
        """
        Initialize a graph from a sequence of nodes.

        Args:
            nodes: Optional sequence of nodes to initialize graph with
        """
        self._nodes: dict[int, Node] = {}
        self._input_nodes: list[Node] = []
        self._output_nodes: list[Node] = []
        self._next_node_id: int = 0
        self._topological_order_cache: list[Node] | None = None
        self._reverse_topological_order_cache: list[Node] | None = None

        if nodes:
            for node in nodes:
                self.add_node(node)

    @property
    def nodes(self) -> list[Node]:
        """All nodes in the graph, ordered by ID."""
        return sorted(self._nodes.values(), key=lambda n: n.id)

    @property
    def input_nodes(self) -> list[Node]:
        """Graph input nodes."""
        return self._input_nodes.copy()

    @property
    def output_nodes(self) -> list[Node]:
        """Graph output nodes."""
        return self._output_nodes.copy()

    @property
    def num_nodes(self) -> int:
        """Total number of nodes in graph."""
        return len(self._nodes)

    @property
    def num_inputs(self) -> int:
        """Number of input nodes."""
        return len(self._input_nodes)

    @property
    def num_outputs(self) -> int:
        """Number of output nodes."""
        return len(self._output_nodes)

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
            node: Node to add

        Raises:
            ValueError: If node ID already exists in graph
        """
        if node.id in self._nodes:
            raise ValueError(f"Node with ID {node.id} already exists in graph")

        self._nodes[node.id] = node

        # Update input/output tracking
        if node.is_input:
            self._input_nodes.append(node)

        # Output nodes are marked explicitly (or inferred later)
        # We'll need to call mark_outputs() after graph construction

        # Track next ID for new nodes
        if node.id >= self._next_node_id:
            self._next_node_id = node.id + 1

        # Invalidate cached orderings
        self._invalidate_caches()

    def get_node(self, node_id: int) -> Node:
        """
        Get a node by its ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            The node with the specified ID

        Raises:
            KeyError: If node ID not found
        """
        return self._nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        """Check if a node with the given ID exists."""
        return node_id in self._nodes

    def mark_outputs(self, output_nodes: Sequence[Node]) -> None:
        """
        Mark specific nodes as graph outputs.

        Args:
            output_nodes: Nodes to mark as outputs

        Raises:
            ValueError: If any node is not in the graph
        """
        for node in output_nodes:
            if node.id not in self._nodes:
                raise ValueError(f"Cannot mark node {node.id} as output: not in graph")

        self._output_nodes = list(output_nodes)

    def infer_outputs(self) -> None:
        """
        Automatically infer output nodes as those with no downstream consumers.

        An output node is one whose result is not used by any other node.
        """
        # Count how many times each node is used as input
        input_counts: dict[int, int] = dict.fromkeys(self._nodes, 0)

        for node in self._nodes.values():
            for input_node in node.inputs:
                input_counts[input_node.id] += 1

        # Nodes with zero downstream users are outputs
        self._output_nodes = [node for node in self._nodes.values() if input_counts[node.id] == 0 and not node.is_input]

    def topological_order(self) -> list[Node]:
        """
        Get nodes in topological order (inputs → outputs).

        Uses Kahn's algorithm for topological sorting. Result is cached.

        Returns:
            List of nodes in topological order

        Raises:
            ValueError: If graph contains cycles
        """
        if self._topological_order_cache is not None:
            return self._topological_order_cache

        # Build in-degree map
        in_degree: dict[int, int] = dict.fromkeys(self._nodes, 0)
        for node in self._nodes.values():
            for _input_node in node.inputs:
                in_degree[node.id] += 1

        # Kahn's algorithm
        queue = deque([node for node in self._nodes.values() if in_degree[node.id] == 0])
        result: list[Node] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Find all nodes that use this node as input
            for dependent_node in self._nodes.values():
                if node in dependent_node.inputs:
                    in_degree[dependent_node.id] -= 1
                    if in_degree[dependent_node.id] == 0:
                        queue.append(dependent_node)

        # Check for cycles
        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles - cannot compute topological order")

        self._topological_order_cache = result
        return result

    def reverse_topological_order(self) -> list[Node]:
        """
        Get nodes in reverse topological order (outputs → inputs).

        Useful for backward propagation algorithms.

        Returns:
            List of nodes in reverse topological order
        """
        if self._reverse_topological_order_cache is not None:
            return self._reverse_topological_order_cache

        self._reverse_topological_order_cache = list(reversed(self.topological_order()))
        return self._reverse_topological_order_cache

    def validate(self) -> None:
        """
        Validate graph structure and consistency.

        Checks:
        - All node inputs reference nodes in the graph
        - No cycles exist (can compute topological order)
        - Input nodes have no inputs
        - At least one input and one output

        Raises:
            ValueError: If validation fails
        """
        # Check we have inputs and outputs
        if not self._input_nodes:
            raise ValueError("Graph has no input nodes")

        if not self._output_nodes:
            # Try to infer outputs
            self.infer_outputs()
            if not self._output_nodes:
                raise ValueError("Graph has no output nodes")

        # Validate all nodes
        for node in self._nodes.values():
            # Input nodes should have no inputs
            if node.is_input and node.inputs:
                raise ValueError(f"Input node {node.id} has inputs")

            # Check all input nodes are in graph
            for input_node in node.inputs:
                if input_node.id not in self._nodes:
                    raise ValueError(f"Node {node.id} references input node {input_node.id} which is not in graph")

        # Check for cycles by computing topological order
        try:
            self.topological_order()
        except ValueError as e:
            raise ValueError(f"Graph validation failed: {e}") from e

    def get_dependencies(self, node: Node, recursive: bool = False) -> set[Node]:
        """
        Get all nodes that this node depends on (directly or recursively).

        Args:
            node: Node to get dependencies for
            recursive: If True, get transitive closure of dependencies

        Returns:
            Set of dependency nodes
        """
        if not recursive:
            return set(node.inputs)

        # BFS to collect all transitive dependencies
        dependencies: set[Node] = set()
        queue = deque(node.inputs)

        while queue:
            dep_node = queue.popleft()
            if dep_node not in dependencies:
                dependencies.add(dep_node)
                queue.extend(dep_node.inputs)

        return dependencies

    def get_dependents(self, node: Node, recursive: bool = False) -> set[Node]:
        """
        Get all nodes that depend on this node (directly or recursively).

        Args:
            node: Node to get dependents for
            recursive: If True, get transitive closure of dependents

        Returns:
            Set of dependent nodes
        """
        # Find direct dependents
        direct_dependents = {n for n in self._nodes.values() if node in n.inputs}

        if not recursive:
            return direct_dependents

        # BFS to collect all transitive dependents
        dependents: set[Node] = set()
        queue = deque(direct_dependents)

        while queue:
            dep_node = queue.popleft()
            if dep_node not in dependents:
                dependents.add(dep_node)
                # Find nodes that depend on dep_node
                next_deps = {n for n in self._nodes.values() if dep_node in n.inputs}
                queue.extend(next_deps)

        return dependents

    def _invalidate_caches(self) -> None:
        """Invalidate cached computations (topological order, etc.)."""
        self._topological_order_cache = None
        self._reverse_topological_order_cache = None

    def __len__(self) -> int:
        """Number of nodes in graph."""
        return len(self._nodes)

    def __contains__(self, node_or_id: Node | int) -> bool:
        """Check if node (or node ID) is in graph."""
        if isinstance(node_or_id, Node):
            return node_or_id.id in self._nodes
        return node_or_id in self._nodes

    def __iter__(self):
        """Iterate over nodes in topological order."""
        return iter(self.topological_order())

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"Graph({self.num_nodes} nodes, {self.num_inputs} inputs, {self.num_outputs} outputs)"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        nodes_str = ", ".join(str(n.id) for n in self.nodes[:5])
        if len(self.nodes) > 5:
            nodes_str += ", ..."
        return f"Graph(nodes=[{nodes_str}], inputs={[n.id for n in self._input_nodes]}, outputs={[n.id for n in self._output_nodes]})"
