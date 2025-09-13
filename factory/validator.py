import ast
from typing import Dict, Set, List
from pydantic import ValidationError

from .schema_models import ConversationFlowOut, EdgeOut

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


class FlowValidationError(Exception):
    pass


class FunctionNodeValidationError(FlowValidationError):
    pass


class SchemaValidationError(FlowValidationError):
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
        
    # Validate function nodes
    _validate_function_nodes(flow, strict)

    # Build adjacency
    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for e in flow.edges:
        if e.from_node_id not in node_ids:
            raise FlowValidationError(f"Edge {e.id} from_node_id not in nodes")
        if e.to_node_id is not None and e.to_node_id not in node_ids:
            raise FlowValidationError(f"Edge {e.id} to_node_id not in nodes")
        if e.type == "prompt" and e.to_node_id is None:
            raise FlowValidationError(f"Edge {e.id} prompt edge requires to_node_id")
        if e.to_node_id is not None:
            adj[e.from_node_id].add(e.to_node_id)

    if not _toposort(node_ids, adj):
        raise FlowValidationError("Flow contains a cycle; DAG required")

    # At least one terminal (skip to None) or explicit terminal node implied by no outgoing edges
    has_terminal = any((e.type == "skip" and e.to_node_id is None) for e in flow.edges)
    if not has_terminal:
        # Accept if exists a node with zero outgoing edges
        if not any(len(adj[n]) == 0 for n in node_ids):
            raise FlowValidationError("No terminal path found (skip-to-null or sink node)")


def _validate_function_nodes(flow: ConversationFlowOut, strict: bool) -> None:
    """Validate function nodes have proper configuration"""
    for node in flow.nodes:
        if node.type == "function":
            if not node.function:
                raise FunctionNodeValidationError(f"Function node {node.id} missing function configuration")
            
            # Validate function type
            valid_types = ["sms", "call_transfer", "rest_webhook"]
            if node.function.function_type not in valid_types:
                raise FunctionNodeValidationError(f"Function node {node.id} has invalid function_type: {node.function.function_type}")
            
            # Basic schema validation if available
            if node.function.parameters_schema and HAS_JSONSCHEMA and strict:
                try:
                    # Basic schema structure validation
                    if not isinstance(node.function.parameters_schema, dict):
                        raise SchemaValidationError(f"Function node {node.id} parameters_schema must be a dict")
                    
                    # Check for required fields based on function type
                    if node.function.function_type == "sms":
                        schema_props = node.function.parameters_schema.get("properties", {})
                        if "to" not in schema_props or "body" not in schema_props:
                            raise SchemaValidationError(f"SMS function node {node.id} schema missing required 'to' or 'body' properties")
                            
                except Exception as e:
                    if strict:
                        raise SchemaValidationError(f"Function node {node.id} schema validation failed: {e}")


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


