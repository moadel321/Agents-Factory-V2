import ast
from typing import Dict, Set, List
from pydantic import ValidationError

from .schema_models import ConversationFlowOut


class FlowValidationError(Exception):
    pass


def _toposort(nodes: Set[str], edges: Dict[str, Set[str]]) -> bool:
    """Return True if acyclic, False otherwise."""
    in_deg: Dict[str, int] = {n: 0 for n in nodes}
    for src, tos in edges.items():
        for t in tos:
            in_deg[t] += 1
    q = [n for n, d in in_deg.items() if d == 0]
    seen = 0
    while q:
        n = q.pop()
        seen += 1
        for t in edges.get(n, set()):
            in_deg[t] -= 1
            if in_deg[t] == 0:
                q.append(t)
    return seen == len(nodes)


def validate_flow(flow: ConversationFlowOut, strict: bool = True) -> None:
    """
    Minimal structural validation of flow definition.
    Invariants like settings presence are pushed to template assertions.

    Args:
        flow: Flow definition to validate
        strict: Unused, kept for API compatibility
    """
    try:
        flow.model_dump()
    except ValidationError as e:
        raise FlowValidationError(str(e))

    node_ids = {n.id for n in flow.nodes}
    if len(node_ids) != len(flow.nodes):
        raise FlowValidationError("Duplicate node ids found")

    if len(flow.nodes) == 0:
        raise FlowValidationError("Flow must have at least one node")

    if not flow.start_node_id:
        raise FlowValidationError("start_node_id must be set")
    if flow.start_node_id not in node_ids:
        raise FlowValidationError("start_node_id not found in nodes")

    # Validate self-loop restrictions
    _validate_self_loop_restrictions(flow)

    # Build adjacency for cycle detection (excluding allowed self-loops)
    node_types = {n.id: n.type for n in flow.nodes}
    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for e in flow.edges:
        if e.from_node_id not in node_ids:
            raise FlowValidationError(f"Edge {e.id} from_node_id not in nodes")
        if e.to_node_id is not None and e.to_node_id not in node_ids:
            raise FlowValidationError(f"Edge {e.id} to_node_id not in nodes")
        if e.type == "prompt" and e.to_node_id is None:
            raise FlowValidationError(f"Edge {e.id} prompt edge requires to_node_id")
        if e.to_node_id is not None:
            # Skip self-loops for conversation nodes in cycle detection
            is_self_loop = e.from_node_id == e.to_node_id
            is_conversation_node = node_types.get(e.from_node_id) == "conversation"
            if not (is_self_loop and is_conversation_node):
                adj[e.from_node_id].add(e.to_node_id)

    if not _toposort(node_ids, adj):
        raise FlowValidationError("Flow contains a cycle; DAG required")

    # At least one terminal (skip to None) or explicit terminal node implied by no outgoing edges
    has_terminal = any((e.type == "skip" and e.to_node_id is None) for e in flow.edges)
    if not has_terminal:
        # Accept if exists a node with zero outgoing edges
        if not any(len(adj[n]) == 0 for n in node_ids):
            raise FlowValidationError("No terminal path found (skip-to-null or sink node)")


def _validate_self_loop_restrictions(flow: ConversationFlowOut) -> None:
    """Validate that only conversation nodes can have self-loops"""
    node_types = {n.id: n.type for n in flow.nodes}

    for edge in flow.edges:
        if edge.from_node_id == edge.to_node_id:  # Self-loop detected
            node_type = node_types.get(edge.from_node_id)
            if node_type == "function":
                raise FlowValidationError(f"Function node {edge.from_node_id} cannot have self-loop (edge {edge.id})")
            elif node_type != "conversation":
                raise FlowValidationError(f"Only conversation nodes can have self-loops, found on {node_type} node {edge.from_node_id} (edge {edge.id})")


def validate_generated_code(code: str) -> List[str]:
    """Validate generated code for syntax and required elements"""
    errors = []
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return errors  # Can't check other things if syntax is broken
    
    # Check for required elements
    required_elements = ["FlowState", "BaseFlowAgent", "entrypoint", "prewarm"]
    for element in required_elements:
        if element not in code:
            errors.append(f"Missing required element: {element}")
    
    # Check for basic imports
    required_imports = ["livekit.agents", "livekit.plugins"]
    for imp in required_imports:
        if imp not in code:
            errors.append(f"Missing required import: {imp}")
    
    return errors


