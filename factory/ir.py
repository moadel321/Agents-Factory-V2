from dataclasses import dataclass
from typing import Any, Dict, List
import unicodedata

from .schema_models import ConversationFlowOut


@dataclass
class IRFlow:
    url_id: str
    name: str
    instructions: str
    stt_provider: str
    llm: Any
    tts: Any
    call_settings: Any
    nodes: List[Any]
    start_class_name: str


def _classify(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    parts = [p for p in safe.split("_") if p]
    return "".join(s.capitalize() for s in parts) + "Agent"


def _ascii_slug(name: str) -> str:
    """Convert arbitrary text to an ASCII-safe slug using [a-zA-Z0-9_-]."""
    # Normalize and strip accents/diacritics
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    # Keep only alnum, dash, underscore; map others to underscore
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in ascii_str)
    parts = [p for p in safe.split("_") if p]
    slug = "_".join(parts)
    # Ensure slug does not start with a digit (OpenAI tools name pattern)
    if slug and slug[0].isdigit():
        slug = f"n_{slug}"
    return slug


def _toolify(name: str) -> str:
    slug = _ascii_slug(name)
    # Ensure valid Python identifier by replacing hyphens with underscores
    slug = (slug or "tool").replace('-', '_')
    tool_name = "go_" + slug
    return _cap_identifier(tool_name)


def _cap_identifier(identifier: str, max_len: int = 64) -> str:
    """Cap identifier length using slicing with a short hash suffix to avoid collisions.

    Keeps it simple and deterministic. If the identifier is already short enough,
    return as-is. Otherwise, slice and append _xxxx (4 hex chars).
    """
    if len(identifier) <= max_len:
        return identifier
    import hashlib
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:4]
    # leave room for underscore + 4 hex chars
    head = identifier[: max_len - 5]
    return f"{head}_{digest}"


def build_ir(flow: ConversationFlowOut) -> IRFlow:
    id_to_node = {n.id: n for n in flow.nodes}
    out_edges = {n.id: [] for n in flow.nodes}
    for e in flow.edges:
        # Include ALL edges, including terminal ones (to_node_id=None)
        out_edges[e.from_node_id].append(e)

    nodes_ir: List[Dict[str, Any]] = []
    for n in flow.nodes:
        class_name = _classify(n.name or n.id)
        node_ir: Dict[str, Any] = {
            "id": n.id,
            "name": n.name,
            "class_name": class_name,
            "type": n.type,
            "out_edges": [],
        }

        # Handle type-specific settings
        if n.type == "conversation":
            node_ir.update({
                # Always carry text and type; semantics handled in template
                "on_enter_text": n.settings.on_enter_text,
                "on_enter_type": n.settings.on_enter_type,
                "skip_response": n.settings.skip_response,
                "capture": [
                    {
                        "name": f.name,
                        "type": getattr(f, "type", "string"),
                        "enum": getattr(f, "enum", None),
                        "multi": getattr(f, "multi", False),
                        "required": getattr(f, "required", False),
                        "description": getattr(f, "description", None),
                    }
                    for f in (n.settings.capture or [])
                ],
            })
        elif n.type == "function":
            node_ir.update({
                "url": n.settings.url,
                "method": n.settings.method,
                "headers": n.settings.headers or {},
                "body": n.settings.body,
                "timeout_ms": n.settings.timeout_ms,
                "retries": n.settings.retries,
                # Optional behavior controls
                "wait_for_result": getattr(n.settings, "wait_for_result", True),
                "speak_during_execution": (
                    {
                        "mode": getattr(getattr(n.settings, "speak_during_execution", None), "mode", None),
                        "text": getattr(getattr(n.settings, "speak_during_execution", None), "text", None),
                        "instructions": getattr(getattr(n.settings, "speak_during_execution", None), "instructions", None),
                    }
                    if getattr(n.settings, "speak_during_execution", None) is not None
                    else None
                ),
            })
        for e in out_edges.get(n.id, []):
            if e.to_node_id is not None:
                # Regular edge to another node
                next_node = id_to_node[e.to_node_id]
                base_label = (e.settings.name if e.settings and e.settings.name else next_node.name) or next_node.id
                # Build ASCII-safe tool name; if empty after slug, fallback to edge id + to_node id
                slug = _ascii_slug(base_label)
                if not slug:
                    slug = _ascii_slug(f"{getattr(e, 'id', 'edge')}_{next_node.id}")
                # Replace hyphens to keep Python method name valid
                slug = slug.replace('-', '_')
                tool_name = _cap_identifier("go_" + slug)
                description = (e.settings.prompt if e.settings else f"Go to {next_node.name}")
                next_class_name = _classify(next_node.name or next_node.id)
            else:
                # Terminal edge (to_node_id=None)
                tool_name = "end_conversation"
                description = (e.settings.prompt if e.settings else "End the conversation")
                next_class_name = None
                
            node_ir["out_edges"].append({
                # Execution/tooling
                "tool_name": tool_name,
                "description": description,
                "next_class_name": next_class_name,
                # Debug/trace metadata
                "edge_id": getattr(e, "id", None),
                "edge_type": getattr(e, "type", None),
                "from_node_id": getattr(e, "from_node_id", n.id),
                "to_node_id": getattr(e, "to_node_id", None),
            })
        nodes_ir.append(node_ir)

    start_class_name = _classify(id_to_node[flow.start_node_id].name or flow.start_node_id) if flow.start_node_id else _classify(flow.nodes[0].name or flow.nodes[0].id)

    return IRFlow(
        url_id=flow.url_id,
        name=flow.name,
        instructions=flow.instructions or "",
        stt_provider=flow.stt_settings.provider,
        llm={
            "model": flow.llm_settings.model,
            "temperature": flow.llm_settings.temperature,
            "max_tokens": flow.llm_settings.max_tokens,
            "provider": flow.llm_settings.provider,
        },
        tts={
            "model": getattr(flow.tts_settings, "model", "eleven_multilingual_v2"),
            "voice_id": getattr(flow.tts_settings, "voice_id", None),
            "provider": getattr(flow.tts_settings, "tts_provider", None),
        },
        call_settings=getattr(flow, "call_settings", None),
        nodes=nodes_ir,
        start_class_name=start_class_name,
    )


