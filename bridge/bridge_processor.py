"""
bridge/bridge_processor.py

The interface layer between content (GNN) and form (Lean).

Responsibilities:
  1. Apply the confidence threshold — deciding which GNN proposals
     are strong enough to present to the formal verifier.
  2. Format the surviving proposals as lean_input.json.
  3. Report the full score distribution so the threshold choice is visible.

What it does NOT do:
  - Define any rules (those are universal laws in form/Laws.lean)
  - Know anything about graph topology (that is the GNN's job)
  - Make any structural judgements (that is Lean's job)
"""

import json
import sys
from pathlib import Path

PREDICTIONS_PATH = Path(__file__).parent / "predictions.json"
LEAN_INPUT_PATH  = Path(__file__).parent / "lean_input.json"
DEFAULT_THRESHOLD = 0.7


def load_predictions(path: Path = PREDICTIONS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_threshold(predictions: list[dict], threshold: float) -> list[dict]:
    return [p for p in predictions if p["confidence"] >= threshold]


def score_band(s: float) -> str:
    if s >= 0.9: return "high    (>=0.9)"
    if s >= 0.7: return "medium  (>=0.7)"
    if s >= 0.5: return "low     (>=0.5)"
    return              "reject  (<0.5) "


def report(data: dict, promoted: list[dict], threshold: float):
    preds = data.get("predictions", [])
    print("── bridge: score distribution ──")
    for p in sorted(preds, key=lambda x: x["confidence"], reverse=True):
        marker = " ← promoted to Lean" if p["confidence"] >= threshold else ""
        print(f"   ({p['source']:>2}, {p['target']:>2})  "
              f"{p['confidence']:.4f}  {score_band(p['confidence'])}{marker}")
    print(f"   {len(promoted)}/{len(preds)} promoted at threshold {threshold}")


def build_lean_input(data: dict, promoted: list[dict], threshold: float) -> dict:
    return {
        "vertex_count":   data["vertex_count"],
        "base_edges":     data["base_edges"],
        "threshold":      threshold,
        "promoted_edges": promoted,
        "all_predictions": data["predictions"],
    }


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_THRESHOLD
    data      = load_predictions()
    promoted  = apply_threshold(data.get("predictions", []), threshold)

    report(data, promoted, threshold)

    lean_input = build_lean_input(data, promoted, threshold)
    with open(LEAN_INPUT_PATH, "w") as f:
        json.dump(lean_input, f, indent=2)
    print(f"   lean_input.json written ({len(promoted)} promoted edges)")


if __name__ == "__main__":
    main()
