import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from factory.schema_models import ConversationFlowOut
from factory.validator import validate_flow
from factory.ir import build_ir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a LiveKit agent from a flow JSON")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    flow_data = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    flow = ConversationFlowOut.model_validate(flow_data)
    validate_flow(flow)

    ir = build_ir(flow)

    # Reuse the factory template for a single source of truth
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "factory" / "templates")),
        autoescape=select_autoescape(disabled_extensions=(".jinja2",)),
        trim_blocks=False,
        lstrip_blocks=False,
    )
    tmpl = env.get_template("agent.jinja2")
    rendered = tmpl.render(flow=ir)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


