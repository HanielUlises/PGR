#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="python3"
LAKE="lake"
THRESHOLD="${THRESHOLD:-0.7}"

log() { printf '\033[1;36m[pipeline]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

check_deps() {
    log "checking dependencies"
    command -v "$PYTHON" >/dev/null || err "python3 not found"
    command -v "$LAKE"   >/dev/null || err "lake not found — install elan + lean4"
    $PYTHON -c "import torch, torch_geometric" 2>/dev/null \
        || err "torch or torch_geometric not installed — pip install torch torch_geometric"
}

step_gnn() {
    log "step 1/4: content — GNN training + inference"
    cd "$ROOT/inference"
    $PYTHON gnn_model.py
    cd "$ROOT"
}

step_bridge() {
    log "step 2/4: bridge — applying threshold=$THRESHOLD"
    cd "$ROOT/bridge"
    $PYTHON bridge_processor.py "$THRESHOLD"
    cd "$ROOT"
}

step_lean() {
    log "step 3/4: form — lake build"
    cd "$ROOT"
    $LAKE build morpheus
}

step_verify() {
    log "step 4/4: form — structural verification"
    cd "$ROOT"
    $LAKE env lean reasoning/GraphVerifier.lean -- bridge/lean_input.json
}

usage() {
    echo "usage: $0 [all|gnn|bridge|lean|verify]"
    echo ""
    echo "  all    — run full pipeline (default)"
    echo "  gnn    — step 1: GNN inference only"
    echo "  bridge — step 2: bridge processor only"
    echo "  lean   — step 3: lake build only"
    echo "  verify — step 4: lean verifier only"
    echo ""
    echo "environment:"
    echo "  THRESHOLD=0.7  confidence threshold for symbolic commitment (default: 0.7)"
    echo ""
    echo "example:"
    echo "  THRESHOLD=0.6 $0 all"
}

check_deps

case "${1:-all}" in
    all)       step_gnn; step_bridge; step_lean; step_verify ;;
    gnn)       step_gnn ;;
    bridge)    step_bridge ;;
    lean)      step_lean ;;
    verify)    step_verify ;;
    -h|--help) usage ;;
    *)         usage; exit 1 ;;
esac

log "done"