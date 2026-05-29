"""
bridge_processor.py

Translates GNN output (facts_and_rules.json) into lean_input.json.

The GNN writes ALL predictions with their confidence scores.
This module applies the threshold, formats the output for Lean,
and reports the full score distribution so the threshold choice
is visible and auditable.
"""

import json
import sys
from pathlib import Path

BRIDGE_PATH = Path(__file__).parent.parent / "bridge" / "facts_and_rules.json"
DEFAULT_THRESHOLD = 0.7


def load_bridge(path: Path = BRIDGE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def format_lean_edge_list(predictions: list[dict]) -> str:
    """Format high-confidence predictions as a Lean array literal."""
    if not predictions:
        return "#[]"
    entries = ", ".join(
        f"({p['source']}, {p['target']}, {p['confidence']})"
        for p in predictions
    )
    return f"#[{entries}]"


def score_band(score: float) -> str:
    """Bucket a confidence score for reporting."""
    if score >= 0.9:  return "high    (≥0.9)"
    if score >= 0.7:  return "medium  (≥0.7)"
    if score >= 0.5:  return "low     (≥0.5)"
    return                    "reject  (<0.5)"


def build_lean_input(data: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    g     = data["graph"]
    preds = data.get("gnn_predictions", [])
    rules = data.get("rules", [])

    # Apply threshold here — the GNN wrote all scores
    high_conf = [p for p in preds if p.get("confidence", 0.0) >= threshold]

    return {
        "vertex_count": g["vertex_count"],
        "edges": g["edges"],
        "threshold": threshold,
        "high_confidence_predictions": high_conf,
        "rule_names": [r["name"] for r in rules],
        "rule_count": len(rules),
        "lean_edge_literal": format_lean_edge_list(high_conf),
        # Pass full prediction list so Lean verifier can log score distribution
        "all_predictions": preds,
    }


def report(data: dict, lean_input: dict):
    preds = data.get("gnn_predictions", [])
    threshold = lean_input["threshold"]

    print("=== Bridge Processor Report ===")
    print(f"vertices         : {lean_input['vertex_count']}")
    print(f"base edges       : {len(lean_input['edges'])}")
    print(f"rules loaded     : {lean_input['rule_count']}  {lean_input['rule_names']}")
    print(f"threshold        : {threshold}")
    print(f"total predictions: {len(preds)}")
    print(f"above threshold  : {len(lean_input['high_confidence_predictions'])}")

    if preds:
        print("\nscore distribution:")
        for p in sorted(preds, key=lambda x: x["confidence"], reverse=True):
            band = score_band(p["confidence"])
            marker = " ← forwarded to Lean" if p.get("above_threshold") else ""
            print(f"  ({p['source']:>2}, {p['target']:>2})  {p['confidence']:.4f}  {band}{marker}")

    print(f"\nlean literal     : {lean_input['lean_edge_literal']}")


def export_lean_json(lean_input: dict, out_path: Path | None = None):
    target = out_path or (BRIDGE_PATH.parent / "lean_input.json")
    with open(target, "w") as f:
        json.dump(lean_input, f, indent=2)
    print(f"\nlean_input written → {target}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    data = load_bridge()
    lean_input = build_lean_input(data, threshold=threshold)
    report(data, lean_input)
    export_lean_json(lean_input, path)


if __name__ == "__main__":
    main()
