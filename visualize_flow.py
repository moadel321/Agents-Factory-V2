#!/usr/bin/env python3
"""
Visualize a Conversation Flow JSON as a Mermaid diagram and HTML viewer.

Always writes two outputs next to the chosen base path:
  - <base>.mmd   (Mermaid source)
  - <base>.html  (self-contained viewer using Mermaid from CDN)

Usage:
  # Write pizza_flow.mmd and pizza_flow.html next to the JSON
  python visualize_flow.py --input examples/flows/pizza_flow.json

  # Choose an explicit base path (creates my_flow.mmd and my_flow.html)
  python visualize_flow.py --input examples/flows/pizza_flow.json --output my_flow

  # Adjust layout direction (TB|LR|BT|RL). TB tends to be less wide.
  python visualize_flow.py --input examples/flows/pizza_flow.json --direction TB
"""
import argparse
import json
import html as htmlmod
from pathlib import Path
from typing import Optional

 
# Prefer using the project's schema models for robust parsing/validation
from factory.schema_models import ConversationFlowOut


def load_flow(flow_path: str) -> ConversationFlowOut:
    """Load and validate the flow JSON into a ConversationFlowOut model."""
    flow_data = json.loads(Path(flow_path).read_text(encoding="utf-8"))
    # Pydantic v2 models support model_validate; standard init works as well
    try:
        flow = ConversationFlowOut.model_validate(flow_data)  # type: ignore[attr-defined]
    except Exception:
        flow = ConversationFlowOut(**flow_data)  # Fallback if running with pydantic v1-like API
    return flow


def _mescape(text: Optional[str]) -> str:
    """Escape text for embedding inside Mermaid labels/blocks."""
    if not text:
        return ""
    return text.replace("\n", " ").replace('"', '\\"')


def build_mermaid(flow: ConversationFlowOut, direction: str = "TB") -> str:
    """Build Mermaid flowchart source from the flow definition."""
    lines = [f"flowchart {direction}"]

    # Legend
    legend = []
    legend.append(f"LLM: {flow.llm_settings.model} (temp={flow.llm_settings.temperature})")
    legend.append(f"STT: {flow.stt_settings.provider} {flow.stt_settings.language}")
    tts_desc = getattr(flow.tts_settings, "model", None) or getattr(flow.tts_settings, "tts_provider", "")
    voice_id = getattr(flow.tts_settings, "voice_id", None)
    legend.append(f"TTS: {tts_desc}{' voice=' + voice_id if voice_id else ''}")
    if flow.call_settings:
        legend.append(
            f"Call: who={flow.call_settings.who_speaks_first}, silence={flow.call_settings.end_call_on_silence_ms}ms, max={flow.call_settings.max_call_duration_ms}ms"
        )
    lines.append("  subgraph Legend")
    joined_legend = "\n".join(legend)
    lines.append(f"    legend[\"{_mescape(joined_legend)}\"]")
    lines.append("  end")

    # Conversation nodes
    conv_nodes = [n for n in flow.nodes if n.type == "conversation"]
    if conv_nodes:
        lines.append("  subgraph Conversation")
        for n in conv_nodes:
            title = f"{n.name} ({n.id})"
            lines.append(f"    {n.id}[\"{_mescape(title)}\"]")
        lines.append("  end")

    # Function nodes
    func_nodes = [n for n in flow.nodes if n.type == "function"]
    if func_nodes:
        lines.append("  subgraph Function")
        for n in func_nodes:
            title = f"{n.name} ({n.id})"
            lines.append(f"    {n.id}(\"{_mescape(title)}\")")
        lines.append("  end")

    # END node if any terminal edges
    if any(e.to_node_id is None for e in flow.edges):
        lines.append("  END[END]")

    # Edges with labels
    for e in flow.edges:
        src = e.from_node_id
        dst = e.to_node_id if e.to_node_id is not None else "END"
        prompt = e.settings.prompt if e.settings else ""
        name = getattr(e.settings, "name", None) if e.settings else None
        label_parts = []
        if name:
            label_parts.append(name)
        if prompt:
            label_parts.append(prompt)
        if not label_parts and e.to_node_id is None:
            label_parts.append("(terminal)")
        label = " — ".join(label_parts)
        if label:
            lines.append(f"  {src} -- \"{_mescape(label)}\" --> {dst}")
        else:
            lines.append(f"  {src} --> {dst}")

    return "\n".join(lines) + "\n"


def write_mermaid_html(mermaid_text: str, out_path: Path) -> None:
    """Write a minimal standalone HTML viewer for a Mermaid diagram."""
    escaped = htmlmod.escape(mermaid_text, quote=False)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Mermaid Diagram</title>
  <style>body{{margin:0;padding:16px;background:#fff}} .mermaid{{font-family:ui-sans-serif,system-ui,sans-serif}}</style>
  <script type=\"module\">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
  <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline';\">
  </head>
<body>
<pre class=\"mermaid\">{escaped}</pre>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _compute_base_output_path(input_path: str, output_hint: Optional[str]) -> Path:
    """Determine base path (without extension) for outputs."""
    if output_hint:
        base = Path(output_hint)
        if base.is_dir():
            return base / Path(input_path).stem
        if base.suffix in {".mmd", ".html"}:
            return base.with_suffix("")
        return base
    return Path(input_path).with_suffix("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a conversation flow JSON as Mermaid + HTML")
    parser.add_argument("--input", "-i", required=True, help="Path to the flow JSON file")
    parser.add_argument("--output", "-o", help="Output base path (creates <base>.mmd and <base>.html)")
    parser.add_argument("--direction", choices=["TB", "LR", "BT", "RL"], default="TB", help="Mermaid layout direction (default: TB)")
    args = parser.parse_args()

    flow = load_flow(args.input)

    base = _compute_base_output_path(args.input, args.output)

    # Generate Mermaid
    mmd_text = build_mermaid(flow, direction=args.direction)
    mmd_path = base.with_suffix(".mmd")
    mmd_path.write_text(mmd_text, encoding="utf-8")
    print(f"Wrote Mermaid file to {mmd_path}")

    # Generate HTML viewer
    html_path = base.with_suffix(".html")
    write_mermaid_html(mmd_text, html_path)
    print(f"Wrote Mermaid HTML viewer to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


