#!/usr/bin/env python3
"""
Visualize a Conversation Flow JSON as Mermaid diagram, HTML viewer, and PNG image.

By default writes three outputs next to the chosen base path:
  - <base>.mmd   (Mermaid source)
  - <base>.html  (self-contained viewer using Mermaid from CDN)
  - <base>.png   (PNG image, requires mmdc CLI)

Usage:
  # Generate all three files (mmd, html, png) next to the JSON
  python visualize_flow.py --input examples/flows/pizza_flow.json

  # Choose an explicit base path
  python visualize_flow.py --input examples/flows/pizza_flow.json --output my_flow

  # Skip PNG generation if mmdc is not installed
  python visualize_flow.py --input examples/flows/pizza_flow.json --no-png

  # Adjust layout direction (TB|LR|BT|RL). TB tends to be less wide.
  python visualize_flow.py --input examples/flows/pizza_flow.json --direction TB

  # Simplify visualization by using dotted lines for hub back-edges
  python visualize_flow.py --input examples/flows/pizza_flow.json --simplify

Requirements:
  - For PNG generation: npm install -g @mermaid-js/mermaid-cli
"""
import argparse
import json
import html as htmlmod
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


# Prefer using the project's schema models for robust parsing/validation
from factory.schema_models import ConversationFlowOut

# Set up logging
logger = logging.getLogger(__name__)


def _check_mmdc_available() -> bool:
    """Check if mermaid-cli (mmdc) is installed and available."""
    return shutil.which("mmdc") is not None


def _generate_png_with_mmdc(
    mmd_path: Path,
    png_path: Path,
    background: str = "white",
    width: int = 1920,
    theme: str = "default"
) -> Path:
    """
    Generate PNG image from Mermaid file using mmdc (mermaid-cli).

    Args:
        mmd_path: Path to the .mmd source file
        png_path: Path where PNG should be saved
        background: Background color (white, transparent, or hex color)
        width: Width of the output image in pixels
        theme: Mermaid theme (default, forest, dark, neutral)

    Returns:
        Path to the generated PNG file

    Raises:
        RuntimeError: If mmdc command fails
    """
    cmd = [
        "mmdc",
        "-i", str(mmd_path),
        "-o", str(png_path),
        "-b", background,
        "-w", str(width),
        "-t", theme
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True  # Required on Windows to find .cmd files
        )

        if result.returncode == 0:
            logger.info(f"Generated PNG: {png_path}")
            return png_path
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"mmdc failed: {error_msg}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("PNG generation timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError("mmdc command not found")


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
    escaped = (
        text.replace("\\", "\\\\")
        .replace("\n", "<br/>")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace('"', '\\"')
    )
    return escaped


def _sanitize_id(node_id: str) -> str:
    """Convert arbitrary node IDs (UUIDs, etc.) into Mermaid-friendly identifiers."""
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", node_id)
    if not sanitized:
        sanitized = "node"
    if sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized


def _short_id(node_id: str) -> str:
    """Return a shortened id for compact display labels."""
    return (node_id or "")[:8] + ("…" if node_id and len(node_id) > 8 else "")


def _wrap_text(text: str, line_len: int) -> str:
    """Wrap long labels at word boundaries to improve readability."""
    if not text:
        return ""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if current else 0)
        if cur_len + add > line_len and current:
            lines.append(" ".join(current))
            current = [w]
            cur_len = len(w)
        else:
            current.append(w)
            cur_len += add
    if current:
        lines.append(" ".join(current))
    return "<br/>".join(lines)


def build_mermaid(
    flow: ConversationFlowOut,
    direction: str = "TB",
    node_line_len: int = 36,
    edge_line_len: int = 40,
    simplify: bool = False,
) -> str:
    """Build Mermaid flowchart source from the flow definition.

    Defaults chosen for readability: LR layout and wrapped labels.
    """
    lines = [f"flowchart {direction}"]

    # Map original node IDs to Mermaid-safe identifiers
    node_id_map: dict[str, str] = {}
    for node in flow.nodes:
        sanitized = _sanitize_id(node.id)
        # Ensure uniqueness in the rare case sanitization collides
        original = sanitized
        counter = 1
        while sanitized in node_id_map.values():
            sanitized = f"{original}_{counter}"
            counter += 1
        node_id_map[node.id] = sanitized

    # Legend
    legend = []
    legend.append(f"Flow: {flow.name}")
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

    # Organize nodes by flow hierarchy
    # Calculate outgoing edge counts for each node
    outgoing_counts = {n.id: 0 for n in flow.nodes}
    incoming_counts = {n.id: 0 for n in flow.nodes}
    for e in flow.edges:
        if e.to_node_id is not None and e.to_node_id in outgoing_counts:
            outgoing_counts[e.from_node_id] += 1
            incoming_counts[e.to_node_id] += 1

    # Categorize nodes
    entry_nodes = [n for n in flow.nodes if n.id == flow.start_node_id]
    terminal_nodes = [n for n in flow.nodes if outgoing_counts[n.id] == 0 and n.id != flow.start_node_id]
    hub_nodes = [n for n in flow.nodes
                 if n.id not in [flow.start_node_id]
                 and n.id not in [tn.id for tn in terminal_nodes]
                 and (incoming_counts[n.id] + outgoing_counts[n.id]) >= 6]  # Hubs have many connections
    service_nodes = [n for n in flow.nodes
                     if n not in entry_nodes
                     and n not in terminal_nodes
                     and n not in hub_nodes]

    # Render nodes in hierarchical subgraphs
    if entry_nodes:
        lines.append("  subgraph \" \"")  # Empty title for cleaner look
        lines.append("    direction TB")
        for n in entry_nodes:
            title_name = _wrap_text(n.name or n.id, node_line_len)
            sid = _short_id(n.id)
            title = f"{title_name} ({sid})"
            mermaid_id = node_id_map[n.id]
            shape = "(" if n.type == "function" else "["
            end_shape = ")" if n.type == "function" else "]"
            lines.append(f"    {mermaid_id}{shape}\"{_mescape(title)}\"{end_shape}")
        lines.append("  end")

    if service_nodes:
        lines.append("  subgraph \" \"")
        lines.append("    direction TB")
        for n in service_nodes:
            title_name = _wrap_text(n.name or n.id, node_line_len)
            sid = _short_id(n.id)
            title = f"{title_name} ({sid})"
            mermaid_id = node_id_map[n.id]
            shape = "(" if n.type == "function" else "["
            end_shape = ")" if n.type == "function" else "]"
            lines.append(f"    {mermaid_id}{shape}\"{_mescape(title)}\"{end_shape}")
        lines.append("  end")

    if hub_nodes:
        lines.append("  subgraph \" \"")
        lines.append("    direction TB")
        for n in hub_nodes:
            title_name = _wrap_text(n.name or n.id, node_line_len)
            sid = _short_id(n.id)
            title = f"{title_name} ({sid})"
            mermaid_id = node_id_map[n.id]
            shape = "(" if n.type == "function" else "["
            end_shape = ")" if n.type == "function" else "]"
            lines.append(f"    {mermaid_id}{shape}\"{_mescape(title)}\"{end_shape}")
        lines.append("  end")

    if terminal_nodes:
        lines.append("  subgraph \" \"")
        lines.append("    direction TB")
        for n in terminal_nodes:
            title_name = _wrap_text(n.name or n.id, node_line_len)
            sid = _short_id(n.id)
            title = f"{title_name} ({sid})"
            mermaid_id = node_id_map[n.id]
            shape = "(" if n.type == "function" else "["
            end_shape = ")" if n.type == "function" else "]"
            lines.append(f"    {mermaid_id}{shape}\"{_mescape(title)}\"{end_shape}")
        lines.append("  end")

    # Start node indicator, if available
    if flow.start_node_id:
        start_id = node_id_map.get(flow.start_node_id, _sanitize_id(flow.start_node_id))
        lines.append(f"  start((Start)) --> {start_id}")

    # END node if any terminal edges
    if any(e.to_node_id is None for e in flow.edges):
        lines.append("  END[END]")

    # Edges with labels
    hub_node_ids = {n.id for n in hub_nodes}
    for e in flow.edges:
        src = node_id_map.get(e.from_node_id)
        if src is None:
            src = _sanitize_id(e.from_node_id)
            node_id_map[e.from_node_id] = src
        if e.to_node_id is not None:
            dst = node_id_map.get(e.to_node_id)
            if dst is None:
                dst = _sanitize_id(e.to_node_id)
                node_id_map[e.to_node_id] = dst
        else:
            dst = "END"
        prompt = e.settings.prompt if e.settings else ""
        name = getattr(e.settings, "name", None) if e.settings else None
        label_parts = []
        if e.type == "skip":
            label_parts.append("(skip)")
        if name:
            label_parts.append(name)
        if prompt:
            label_parts.append(_wrap_text(prompt, edge_line_len))
        if not label_parts and e.to_node_id is None:
            label_parts.append("(terminal)")
        label = " | ".join(label_parts)

        # Use dotted lines for edges going TO hub nodes when simplify is enabled
        is_hub_edge = simplify and e.to_node_id in hub_node_ids
        arrow = "-.->" if is_hub_edge else "-->"

        if label:
            lines.append(f"  {src} -- \"{_mescape(label)}\" {arrow} {dst}")
        else:
            lines.append(f"  {src} {arrow} {dst}")

    # Global styling hint: smoother curves for overlapping edges
    lines.append("  linkStyle default interpolate basis")
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
  <style>
    body{{margin:0;padding:16px;background:#fff}}
    .mermaid{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:14px;}}
  </style>
  <script type=\"module\">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{
      startOnLoad: true,
      securityLevel: 'loose',
      flowchart: {{
        htmlLabels: true,
        curve: 'basis',
        nodeSpacing: 100,
        rankSpacing: 150,
        diagramPadding: 16
      }}
    }});
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


def visualize_flow(
    input_path: str,
    output_path: Optional[str] = None,
    direction: str = "TB",
    node_line_len: int = 36,
    edge_line_len: int = 40,
    simplify: bool = False,
    generate_png: bool = True,
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Generate flow visualization files programmatically.

    Args:
        input_path: Path to the flow JSON file
        output_path: Optional output base path (without extension)
        direction: Mermaid layout direction (TB, LR, BT, RL)
        node_line_len: Max characters per line for node labels
        edge_line_len: Max characters per line for edge labels
        simplify: Style hub edges as dotted for reduced clutter
        generate_png: Generate PNG image using mmdc (default: True)

    Returns:
        Tuple of (mmd_path, html_path, png_path). png_path is None if not generated.
    """
    flow = load_flow(input_path)
    base = _compute_base_output_path(input_path, output_path)

    # Generate Mermaid source
    mmd_text = build_mermaid(flow, direction=direction, node_line_len=node_line_len, edge_line_len=edge_line_len, simplify=simplify)
    mmd_path = base.with_suffix(".mmd")
    mmd_path.write_text(mmd_text, encoding="utf-8")

    # Generate HTML viewer
    html_path = base.with_suffix(".html")
    write_mermaid_html(mmd_text, html_path)

    # Generate PNG image (default enabled)
    png_path = None
    if generate_png:
        if _check_mmdc_available():
            try:
                png_path = _generate_png_with_mmdc(mmd_path, base.with_suffix(".png"))
            except Exception as e:
                logger.warning(f"PNG generation failed: {e}")
        else:
            logger.warning("mmdc not found. Install with: npm install -g @mermaid-js/mermaid-cli")

    return mmd_path, html_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a conversation flow JSON as Mermaid + HTML")
    parser.add_argument("--input", "-i", required=True, help="Path to the flow JSON file")
    parser.add_argument("--output", "-o", help="Output base path (creates <base>.mmd and <base>.html)")
    parser.add_argument("--direction", choices=["TB", "LR", "BT", "RL"], default="TB", help="Mermaid layout direction (default: TB)")
    parser.add_argument("--node-line-len", type=int, default=36, help="Max characters per line for node labels (default: 36)")
    parser.add_argument("--edge-line-len", type=int, default=40, help="Max characters per line for edge labels (default: 40)")
    parser.add_argument("--simplify", action="store_true", help="Style hub back-edges as dotted to reduce visual clutter")
    parser.add_argument("--no-png", action="store_true", dest="no_png", help="Skip PNG generation (requires mmdc: npm install -g @mermaid-js/mermaid-cli)")
    args = parser.parse_args()

    try:
        mmd_path, html_path, png_path = visualize_flow(
            input_path=args.input,
            output_path=args.output,
            direction=args.direction,
            node_line_len=args.node_line_len,
            edge_line_len=args.edge_line_len,
            simplify=args.simplify,
            generate_png=not args.no_png,
        )

        print(f"Wrote Mermaid file to {mmd_path}")
        print(f"Wrote Mermaid HTML viewer to {html_path}")
        if png_path:
            print(f"Wrote PNG image to {png_path}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


