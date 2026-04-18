import json
import sys
from pathlib import Path

BRIDGE_PATH = Path(__file__).parent.parent / "bridge" / "facts_and_rules.json"

def load_bridge() -> dict:
    with open(BRIDGE_PATH) as f:
        return json.load(f)

def format_lean_edge_list(predictions: list[dict]) -> str:
    if not predictions:
        return "#[]"
    entries = ", ".join(
        f"({p['source']}, {p['target']}, {p['confidence']})"
        for p in predictions
    )
    return f"#[{entries}]"

def build_lean_input(data: dict) -> dict:
    g = data["graph"]
    preds = data.get("gnn_predictions", [])
    rules = data.get("rules", [])

    high_conf = [p for p in preds if p.get("confidence", 0) >= 0.7]

    return {
        "vertex_count": g["vertex_count"],
        "edges": g["edges"],
        "high_confidence_predictions": high_conf,
        "rule_names": [r["name"] for r in rules],
        "rule_count": len(rules),
        "lean_edge_literal": format_lean_edge_list(high_conf)
    }

def export_lean_json(lean_input: dict, out_path: Path | None = None):
    target = out_path or (BRIDGE_PATH.parent / "lean_input.json")
    with open(target, "w") as f:
        json.dump(lean_input, f, indent=2)
    print(f"lean_input written → {target}")

def report(lean_input: dict):
    print("=== Bridge Processor Report ===")
    print(f"vertices         : {lean_input['vertex_count']}")
    print(f"base edges       : {len(lean_input['edges'])}")
    print(f"rules loaded     : {lean_input['rule_count']}  {lean_input['rule_names']}")
    print(f"high-conf preds  : {len(lean_input['high_confidence_predictions'])}")
    print("predictions:")
    for p in lean_input["high_confidence_predictions"]:
        u, v, s = p["source"], p["target"], p["confidence"]
        print(f"  ({u}, {v})  confidence={s}")
    print(f"lean literal     : {lean_input['lean_edge_literal']}")

def main():
    data = load_bridge()
    lean_input = build_lean_input(data)
    report(lean_input)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    export_lean_json(lean_input, out)

if __name__ == "__main__":
    main()