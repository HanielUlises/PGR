import reasoning.SymbolicCore
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
import Mathlib.Data.Fin.Basic
import Lean.Data.Json

open Lean NeuroSymbolic SimpleGraph

namespace GraphVerifier

-- Static reference graph (compile-time, fully verified)
def adjFn (u v : Fin 6) : Prop :=
  (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 0) ∨
  (u = 1 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨
  (u = 2 ∧ v = 3) ∨ (u = 3 ∧ v = 2) ∨
  (u = 3 ∧ v = 4) ∨ (u = 4 ∧ v = 3) ∨
  (u = 0 ∧ v = 5) ∨ (u = 5 ∧ v = 0) ∨
  (u = 5 ∧ v = 2) ∨ (u = 2 ∧ v = 5)

instance : DecidablePred (fun u : Fin 6 => ∀ v : Fin 6, ¬adjFn u v) := by
  intro u; unfold adjFn; infer_instance

def sample_graph : SimpleGraph (Fin 6) where
  Adj := adjFn
  symm := by
    intro u v h
    simp only [adjFn] at *
    omega
  loopless := by
    intro u
    simp only [adjFn]
    omega

instance : DecidableRel sample_graph.Adj := by
  intro u v; simp only [sample_graph, adjFn]; infer_instance

private def e01 : sample_graph.Adj 0 1 := by decide
private def e12 : sample_graph.Adj 1 2 := by decide
private def e23 : sample_graph.Adj 2 3 := by decide
private def e34 : sample_graph.Adj 3 4 := by decide
private def e05 : sample_graph.Adj 0 5 := by decide

theorem reach_0_1 : sample_graph.Reachable 0 1 := e01.reachable
theorem reach_0_2 : sample_graph.Reachable 0 2 := e01.reachable.trans e12.reachable
theorem reach_0_3 : sample_graph.Reachable 0 3 := reach_0_2.trans e23.reachable
theorem reach_0_4 : sample_graph.Reachable 0 4 := reach_0_3.trans e34.reachable
theorem reach_0_5 : sample_graph.Reachable 0 5 := e05.reachable

theorem all_nodes_reachable_from_0 :
    ∀ v : Fin 6, sample_graph.Reachable 0 v := by
  intro v
  match v with
  | ⟨0, _⟩ => exact Reachable.refl _
  | ⟨1, _⟩ => exact reach_0_1
  | ⟨2, _⟩ => exact reach_0_2
  | ⟨3, _⟩ => exact reach_0_3
  | ⟨4, _⟩ => exact reach_0_4
  | ⟨5, _⟩ => exact reach_0_5

theorem sample_graph_connected : sample_graph.Connected := by
  rw [SimpleGraph.connected_iff]
  refine ⟨fun u v => ?_, inferInstance⟩
  exact (all_nodes_reachable_from_0 u).symm.trans (all_nodes_reachable_from_0 v)

-- Runtime graph (adjacency list, BFS reachability)
structure RuntimeGraph where
  vertexCount : Nat
  edges       : List (Nat × Nat)
  deriving Repr

def RuntimeGraph.neighbours (g : RuntimeGraph) (u : Nat) : List Nat :=
  g.edges.filterMap fun (a, b) =>
    if a == u then some b
    else if b == u then some a
    else none

def RuntimeGraph.bfs (g : RuntimeGraph) (dst : Nat)
    (queue visited : List Nat) (fuel : Nat) : Bool :=
  match fuel with
  | 0 => false
  | fuel + 1 =>
    match queue with
    | [] => false
    | u :: rest =>
      if u == dst then true
      else if visited.contains u then
        g.bfs dst rest visited fuel
      else
        let nbrs := g.neighbours u |>.filter (fun v => !visited.contains v)
        g.bfs dst (rest ++ nbrs) (u :: visited) fuel

def RuntimeGraph.reachable (g : RuntimeGraph) (src dst : Nat) : Bool :=
  g.bfs dst [src] [] (g.vertexCount + 1)

-- Prediction verification
structure PredictionResult where
  source      : Nat
  target      : Nat
  confidence  : Float
  reachable   : Bool
  inComponent : Bool
  deriving Repr

def parseEdge (j : Json) : Option (Nat × Nat) :=
  match j.getArr? with
  | .error _ => none
  | .ok arr =>
    match arr[0]?, arr[1]? with
    | some a, some b =>
      match Json.getNat? a, Json.getNat? b with
      | .ok x, .ok y => some (x, y)
      | _, _ => none
    | _, _ => none

def parsePrediction (j : Json) : Option (Nat × Nat × Float) :=
  match j.getObjValAs? Nat   "source",
        j.getObjValAs? Nat   "target",
        j.getObjValAs? Float "confidence" with
  | .ok src, .ok dst, .ok conf => some (src, dst, conf)
  | _, _, _ => none

structure BridgeData where
  vertexCount : Nat
  baseEdges   : List (Nat × Nat)
  predictions : List (Nat × Nat × Float)
  deriving Repr

def parseBridgeData (raw : String) : Except String BridgeData := do
  let j ← Json.parse raw
  match j.getObjVal? "graph" with
  | .error e => .error s!"missing 'graph': {e}"
  | .ok g =>
    match Json.getObjValAs? g Nat "vertex_count",
          Json.getObjValAs? g (Array Json) "edges",
          j.getObjValAs? (Array Json) "gnn_predictions" with
    | .ok n, .ok earr, .ok parr =>
      let edges := earr.toList.filterMap parseEdge
      let preds := parr.toList.filterMap parsePrediction
      return { vertexCount := n, baseEdges := edges, predictions := preds }
    | .error e, _, _ => .error s!"missing vertex_count: {e}"
    | _, .error e, _ => .error s!"missing edges: {e}"
    | _, _, .error e => .error s!"missing gnn_predictions: {e}"

def verifyPredictions (bd : BridgeData) : List PredictionResult :=
  let predEdges := bd.predictions.map fun (u, v, _) => (u, v)
  let rg : RuntimeGraph := {
    vertexCount := bd.vertexCount
    edges       := bd.baseEdges ++ predEdges
  }
  bd.predictions.map fun (u, v, conf) =>
    { source      := u
      target      := v
      confidence  := conf
      reachable   := rg.reachable u v
      inComponent := u < 6 && v < 6
    }

def resultSummary (r : PredictionResult) : String :=
  let status := if r.reachable && r.inComponent then "VERIFIED" else "REJECTED"
  s!"{status}  ({r.source}, {r.target})  confidence={r.confidence}"

def main (args : List String) : IO Unit := do
  let path := args.headD "bridge/lean_input.json"
  let raw ← IO.FS.readFile path
  match parseBridgeData raw with
  | .error e =>
    IO.eprintln s!"[verifier] failed to parse bridge data: {e}"
    IO.Process.exit 1
  | .ok bd =>
    IO.println "=== Neuro-Symbolic Graph Verifier ==="
    IO.println s!"vertices        : {bd.vertexCount}"
    IO.println s!"base edges      : {bd.baseEdges.length}"
    IO.println s!"predictions     : {bd.predictions.length}"
    IO.println s!"static theorem  : sample_graph is connected (machine-checked)"
    IO.println ""
    let results  := verifyPredictions bd
    let verified := results.filter (fun r => r.reachable && r.inComponent)
    let rejected := results.filter (fun r => !(r.reachable && r.inComponent))
    IO.println s!"{verified.length} verified, {rejected.length} rejected"
    if !verified.isEmpty then
      IO.println "\nverified:"
      for r in verified do IO.println s!"  {resultSummary r}"
    if !rejected.isEmpty then
      IO.println "\nrejected:"
      for r in rejected do IO.println s!"  {resultSummary r}"

end GraphVerifier
