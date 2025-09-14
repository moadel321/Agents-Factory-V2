#!/usr/bin/env python3
"""
Visualize a Conversation Flow JSON as a Graphviz diagram.

Usage:
  python visualize_flow.py --input examples/pizza_flow.json --output flow.svg

If Graphviz binaries are not available, the script falls back to writing a DOT file.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except Exception:
    HAS_GRAPHVIZ = False

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


def _truncate(text: Optional[str], max_len: int = 120) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def build_graph(flow: ConversationFlowOut, fmt: str = "svg") -> "Digraph":
    """Build a Graphviz Digraph from the flow definition."""
    g = Digraph(
        comment=f"Flow: {flow.name} ({flow.url_id})",
        format=fmt,
        graph_attr={
            "rankdir": "LR",
            "concentrate": "true",
            "fontsize": "11",
            "labelloc": "t",
            "label": f"{flow.name}\\nURL ID: {flow.url_id}",
        },
        node_attr={"fontsize": "10", "style": "filled", "fillcolor": "white"},
        edge_attr={"fontsize": "9"},
    )

    # Legend / flow-level settings
    legend_lines = []
    legend_lines.append(f"LLM: {flow.llm_settings.model} (temp={flow.llm_settings.temperature})")
    legend_lines.append(f"STT: {flow.stt_settings.provider} {flow.stt_settings.language}")
    tts_desc = getattr(flow.tts_settings, "model", None) or getattr(flow.tts_settings, "tts_provider", "")
    voice_id = getattr(flow.tts_settings, "voice_id", None)
    if voice_id:
        legend_lines.append(f"TTS: {tts_desc} voice={voice_id}")
    else:
        legend_lines.append(f"TTS: {tts_desc}")
    if flow.call_settings:
        legend_lines.append(
            f"Call: who={flow.call_settings.who_speaks_first}, silence={flow.call_settings.end_call_on_silence_ms}ms, max={flow.call_settings.max_call_duration_ms}ms"
        )
    if flow.post_call_analysis:
        legend_lines.append(
            f"Post-call analysis: {flow.post_call_analysis.model} ({len(flow.post_call_analysis.analysis_items)} items)"
        )

    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", fontsize="11")
        c.node(
            "legend",
            label="\n".join(legend_lines) or "Flow settings",
            shape="note",
            fillcolor="lightgray",
        )

    # Nodes
    node_id_to_graph_id = {}
    for n in flow.nodes:
        is_conversation = n.type == "conversation"
        is_function = n.type == "function"

        title = f"{n.name}\\n({n.id})"
        if is_conversation:
            details = []
            details.append("Type: conversation")
            enter_text = _truncate(n.settings.on_enter_text)
            if n.settings.on_enter_type == "prompt" and enter_text:
                details.append(f"On enter: {enter_text}")
            if n.settings.llm_overrides:
                ov = n.settings.llm_overrides
                parts = []
                if ov.model:
                    parts.append(f"model={ov.model}")
                if ov.temperature is not None:
                    parts.append(f"temp={ov.temperature}")
                details.append("Overrides: " + ", ".join(parts))
            label = title + ("\\n" + "\\n".join(details) if details else "")
            shape = "box"
            fill = "#E6F2FF"  # light blue
        elif is_function and n.function is not None:
            details = [
                "Type: function",
                f"func={n.function.function_type}",
                f"timeout={n.function.timeout_ms}ms retries={n.function.retries}",
            ]
            label = title + "\\n" + "\\n".join(details)
            shape = "ellipse"
            fill = "#FFF4CC"  # light yellow
        else:
            label = title + "\\nType: unknown"
            shape = "box"
            fill = "white"

        g.node(n.id, label=label, shape=shape, fillcolor=fill)
        node_id_to_graph_id[n.id] = n.id

    # Explicit END node only if needed
    needs_end = any(e.to_node_id is None for e in flow.edges)
    if needs_end:
        g.node("__END__", label="END", shape="doublecircle", fillcolor="#EEEEEE")

    # Edges
    for e in flow.edges:
        src = e.from_node_id
        dst = e.to_node_id if e.to_node_id is not None else "__END__"

        prompt = e.settings.prompt if e.settings else ""
        name = getattr(e.settings, "name", None) if e.settings else None
        label_parts = []
        if name:
            label_parts.append(name)
        if prompt:
            label_parts.append(prompt)
        label = " — ".join(label_parts) if label_parts else e.type

        style = "dashed" if e.type == "skip" else "solid"
        g.edge(src, dst, label=label, style=style)

    return g


def build_dot_string(flow: ConversationFlowOut) -> str:
    """Build a plain DOT string for the flow (fallback when rendering is unavailable)."""
    lines = ["digraph flow {", "rankdir=LR;", "concentrate=true;"]
    lines.append(
        f"label=\"{flow.name}\\nURL ID: {flow.url_id}\"; labelloc=t; fontsize=11;"
    )

    # Nodes
    for n in flow.nodes:
        title = f"{n.name}\\n({n.id})"
        if n.type == "conversation":
            enter_text = _truncate(n.settings.on_enter_text)
            parts = ["Type: conversation"]
            if n.settings.on_enter_type == "prompt" and enter_text:
                parts.append(f"On enter: {enter_text}")
            label = title + ("\\n" + "\\n".join(parts) if parts else "")
            lines.append(f'"{n.id}" [shape=box, style=filled, fillcolor="#E6F2FF", label="{label}"];')
        elif n.type == "function" and n.function is not None:
            parts = [
                "Type: function",
                f"func={n.function.function_type}",
                f"timeout={n.function.timeout_ms}ms retries={n.function.retries}",
            ]
            label = title + "\\n" + "\\n".join(parts)
            lines.append(f'"{n.id}" [shape=ellipse, style=filled, fillcolor="#FFF4CC", label="{label}"];')
        else:
            lines.append(f'"{n.id}" [shape=box, label="{title}\\nType: unknown"];')

    needs_end = any(e.to_node_id is None for e in flow.edges)
    if needs_end:
        lines.append('"__END__" [shape=doublecircle, style=filled, fillcolor="#EEEEEE", label="END"];')

    # Edges
    for e in flow.edges:
        src = e.from_node_id
        dst = e.to_node_id if e.to_node_id is not None else "__END__"
        prompt = e.settings.prompt if e.settings else ""
        name = getattr(e.settings, "name", None) if e.settings else None
        label_parts = []
        if name:
            label_parts.append(name)
        if prompt:
            label_parts.append(prompt)
        label = " — ".join(label_parts) if label_parts else e.type
        style = "dashed" if e.type == "skip" else "solid"
        lines.append(f'"{src}" -> "{dst}" [label="{label}", style={style}];')

    lines.append("}")
    return "\n".join(lines)


def locate_dot_executable() -> Optional[str]:
    """Locate Graphviz dot executable if present."""
    # Env override
    env_dot = os.getenv("GRAPHVIZ_DOT")
    if env_dot and os.path.exists(env_dot):
        return env_dot
    # PATH
    which = shutil.which("dot")
    if which:
        return which
    # Common Windows paths
    candidates = [
        r"C:\\Program Files\\Graphviz\\bin\\dot.exe",
        r"C:\\Program Files (x86)\\Graphviz\\bin\\dot.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a conversation flow JSON with Graphviz")
    parser.add_argument("--input", "-i", required=True, help="Path to the flow JSON file")
    parser.add_argument("--output", "-o", help="Output diagram path (e.g., flow.svg, flow.png, flow.dot)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["svg", "png", "pdf", "dot"],
        default="png",
        help="Output format (default: png)",
    )
    args = parser.parse_args()
    
    # Set Graphviz path if not found in PATH
    if HAS_GRAPHVIZ and not os.getenv("GRAPHVIZ_DOT"):
        graphviz_paths = [
            r"C:\Program Files\Graphviz\bin\dot.exe",
            r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
        ]
        for path in graphviz_paths:
            if os.path.exists(path):
                os.environ["GRAPHVIZ_DOT"] = path
                break

    flow = load_flow(args.input)

    # Determine output path
    out_path = Path(args.output) if args.output else Path(args.input).with_suffix(f".{args.format}")

    # If DOT requested explicitly, write DOT and exit
    if args.format == "dot":
        dot_text = build_dot_string(flow)
        out_path.write_text(dot_text, encoding="utf-8")
        print(f"Wrote DOT file to {out_path}")
        return 0

    # Try rendering with graphviz; if system binaries are missing, fall back to DOT
    try:
        g = build_graph(flow, fmt=args.format)
        # Render to the exact output path; graphviz adds extensions if using render()
        # Use pipe to write directly
        binary = g.pipe(format=args.format)
        out_path.write_bytes(binary)
        print(f"Wrote diagram to {out_path}")
        return 0
    except Exception as e:
        print(f"Python graphviz render failed ({e}); trying system 'dot'.")

    # Fallback: try system dot
    dot_exe = locate_dot_executable()
    if dot_exe:
        dot_text = build_dot_string(flow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dot") as tf:
            tf.write(dot_text.encode("utf-8"))
            temp_dot = tf.name
        try:
            cmd = [dot_exe, f"-T{args.format}", temp_dot, "-o", str(out_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr or proc.stdout)
            print(f"Wrote diagram to {out_path}")
            return 0
        except Exception as e:
            print(f"System 'dot' render failed ({e}).")
        finally:
            try:
                os.unlink(temp_dot)
            except Exception:
                pass

    # Last resort: write DOT next to requested output
    dot_text = build_dot_string(flow)
    fallback_path = out_path.with_suffix(".dot")
    fallback_path.write_text(dot_text, encoding="utf-8")
    print(f"Graphviz not available; wrote DOT file to {fallback_path}. Use 'dot -T{args.format} {fallback_path} -o {out_path}' to render.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


