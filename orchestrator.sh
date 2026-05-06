#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="python3"
LAKE="lake"

log() { printf '\033[1;36m[pipeline]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

check_deps() {
    log "checking dependencies"
    command -v "$PYTHON" >/dev/null || err "python3 not found"
    command -v "$LAKE"   >/dev/null || err "lake not found — install elan + lean4"
    $PYTHON -c "import torch, torch_geometric" 2>/dev/null \
        || err "torch or torch_geometric not installed"
}

step_gnn() {
    log "step 1. running GNN (PyG)"
    cd "$ROOT/inference"
    $PYTHON gnn_model.py
    cd "$ROOT"
}

step_bridge() {
    log "step 2. processing bridge JSON"
    cd "$ROOT/inference"
    $PYTHON bridge_processor.py "$ROOT/bridge/lean_input.json"
    cd "$ROOT"
}

step_lean() {
    log "step 3. building Lean 4 symbolic core"
    cd "$ROOT"
    $LAKE build NeuroSymbolicGraph
}

step_verify() {
    log "step 4. running Lean verifier"
    cd "$ROOT"
    $LAKE env lean reasoning/GraphVerifier.lean
}

usage() {
    echo "usage: $0 [all|gnn|bridge|lean|verify]"
    echo "  all    — run full pipeline (default)"
    echo "  gnn    — only step 1 (GNN inference)"
    echo "  bridge — only step 2 (bridge JSON)"
    echo "  lean   — only step 3 (lake build)"
    echo "  verify — only step 4 (lean eval)"
}

check_deps

case "${1:-all}" in
    all)    step_gnn; step_bridge; step_lean; step_verify ;;
    gnn)    step_gnn ;;
    bridge) step_bridge ;;
    lean)   step_lean ;;
    verify) step_verify ;;
    *)      usage; exit 1 ;;
esac

log "pipeline complete"
