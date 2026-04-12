"""
Computation graph representation and utilities.

A Graph is a directed acyclic graph (DAG) of operations that can be analyzed
with bound propagation methods.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from .node import AbstractValueType, Node


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

    def __init__(self, nodes: list[Node]) -> None:
        """
        Initialize a graph from a sequence of nodes.

        Args:
            nodes: Optional sequence of nodes to initialize graph with
        """
        self._nodes: list[Node] = nodes
        self._input_nodes: list[Node] = [node for node in nodes if node.is_input]
        self._output_nodes: list[Node] = []
        self._next_node_id: int = 0
        self._topological_order_cache: list[Node] | None = None
        self._reverse_topological_order_cache: list[Node] | None = None

    @property
    def nodes(self) -> list[Node]:
        """All nodes in the graph, ordered by ID."""
        return self._nodes

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

    def has_node(self, node_id: int) -> bool:
        """Check if a node with the given ID exists."""
        return any(node.id == node_id for node in self._nodes)

    def mark_outputs(self, output_nodes: Sequence[Node]) -> None:
        """
        Mark specific nodes as graph outputs.

        Args:
            output_nodes: Nodes to mark as outputs

        Raises:
            ValueError: If any node is not in the graph
        """
        node_ids = {node.id for node in self._nodes}
        for node in output_nodes:
            if node.id not in node_ids:
                raise ValueError(f"Cannot mark node {node.id} as output: not in graph")

        self._output_nodes = list(output_nodes)

    def infer_outputs(self) -> None:
        """
        Automatically infer output nodes as those with no downstream consumers.

        An output node is one whose result is not used by any other node.
        """
        # Count how many times each node is used as input
        input_counts: dict[int, int] = {node.id: 0 for node in self._nodes}

        for node in self._nodes:
            for input_node in node.inputs:
                input_counts[input_node.id] += 1

        # Nodes with zero downstream users are outputs
        self._output_nodes = [node for node in self._nodes if input_counts[node.id] == 0 and not node.is_input]

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
        in_degree: dict[int, int] = {node.id: 0 for node in self._nodes}
        for node in self._nodes:
            for _input_node in node.inputs:
                in_degree[node.id] += 1

        # Kahn's algorithm
        queue = deque([node for node in self._nodes if in_degree[node.id] == 0])
        result: list[Node] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Find all nodes that use this node as input
            for dependent_node in self._nodes:
                # Count how many times this node appears in dependent's inputs
                count = dependent_node.inputs.count(node)
                if count > 0:
                    in_degree[dependent_node.id] -= count
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
        for node in self._nodes:
            # Input nodes should have no inputs
            if node.is_input and node.inputs:
                raise ValueError(f"Input node {node.id} has inputs")

            # Check all input nodes are in graph
            for input_node in node.inputs:
                if input_node.id not in [n.id for n in self._nodes]:
                    raise ValueError(f"Node {node.id} references input node {input_node.id} which is not in graph")

        # Check for cycles by computing topological order
        self.topological_order()

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
        direct_dependents = {n for n in self._nodes if node in n.inputs}

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
                next_deps = {n for n in self._nodes if dep_node in n.inputs}
                queue.extend(next_deps)

        return dependents

    def annotate_input_kinds(self):
        """
        Annotate nodes with constant/abstract input kind information.

        Inputs are always treated as abstract. Constant/parameter nodes are
        treated as constant. Operation nodes become constant only if all their
        inputs are constant.

        Returns:
            Mapping from node ID to tuple of input kinds for operation-like nodes.
            Kinds are the strings "constant" or "abstract".
        """
        output_signatures: dict[int, AbstractValueType] = {}
        input_signatures: dict[int, tuple[AbstractValueType, ...]] = {}

        for node in self.topological_order():
            if node.is_input:
                output_signature = AbstractValueType.ABSTRACT
            elif node.is_value:
                output_signature = AbstractValueType.CONSTANT
            else:
                input_kinds: list[AbstractValueType] = []
                for input_node in node.inputs:
                    if input_node.id not in output_signatures:
                        raise ValueError(f"Input node {input_node.id} for node {node.id} has no inferred kind")
                    input_kinds.append(output_signatures[input_node.id])

                signature = tuple(input_kinds)
                input_signatures[node.id] = signature
                node.input_signature = signature

                output_signature = (
                    AbstractValueType.CONSTANT
                    if signature and all(kind == AbstractValueType.CONSTANT for kind in signature)
                    else AbstractValueType.ABSTRACT
                )

            output_signatures[node.id] = output_signature
            node.output_signature = output_signature

    def __len__(self) -> int:
        """Number of nodes in graph."""
        return len(self._nodes)

    def __contains__(self, node_or_id: Node | int) -> bool:
        """Check if node (or node ID) is in graph."""
        if isinstance(node_or_id, Node):
            return node_or_id in self._nodes
        return node_or_id in [node.id for node in self._nodes]

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
