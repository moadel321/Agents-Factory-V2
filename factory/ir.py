from dataclasses import dataclass
from typing import Any, Dict, List

from .schema_models import ConversationFlowOut


@dataclass
class IRFlow:
    url_id: str
    name: str
    instructions: str
    stt_provider: str
    llm: Any
    tts: Any
    post_call_analysis: Any
    call_settings: Any
    nodes: List[Any]
    start_class_name: str


def _classify(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    parts = [p for p in safe.split("_") if p]
    return "".join(s.capitalize() for s in parts) + "Agent"


def _toolify(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    parts = [p for p in safe.split("_") if p]
    return "go_" + "_".join(parts)


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
            "class_name": class_name,
            "type": n.type,
            "instructions": (flow.instructions or "") + "\n\n" + (n.global_settings.prompt if n.global_settings else ""),
            "on_enter_text": n.settings.on_enter_text if n.settings.on_enter_type == "prompt" else None,
            "skip_response": n.settings.skip_response,
            "out_edges": [],
            # Node-level capture fields (optional)
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
        }
        if n.type == "function" and n.function is not None:
            node_ir["function"] = {
                "function_type": n.function.function_type,
                "timeout_ms": getattr(n.function, "timeout_ms", 10000),
                "retries": getattr(n.function, "retries", 0),
                "call_kwargs": {},
            }
        for e in out_edges.get(n.id, []):
            if e.to_node_id is not None:
                # Regular edge to another node
                next_node = id_to_node[e.to_node_id]
                tool_name = _toolify((e.settings.name if e.settings and e.settings.name else next_node.name) or next_node.id)
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
        },
        tts={
            "model": getattr(flow.tts_settings, "model", "eleven_multilingual_v2"),
            "voice_id": getattr(flow.tts_settings, "voice_id", None),
        },
        post_call_analysis=getattr(flow, "post_call_analysis", None),
        call_settings=getattr(flow, "call_settings", None),
        nodes=nodes_ir,
        start_class_name=start_class_name,
    )


