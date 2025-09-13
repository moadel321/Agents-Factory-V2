from typing import List, Dict, Any


def build_router_tool_descriptions(out_edges: List[Dict[str, Any]]) -> str:
    lines = []
    for e in out_edges:
        lines.append(f"- {e['tool_name']}: {e['description']}")
    return "\n".join(lines)


def build_post_call_prompt(analysis_items: List[Dict[str, Any]]) -> str:
    lines = [
        "Return strict JSON with these fields and types:",
    ]
    for item in analysis_items:
        lines.append(f"- {item['name']} ({item['type']})")
    return "\n".join(lines)


