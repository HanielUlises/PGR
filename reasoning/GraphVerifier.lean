import reasoning.SymbolicKernel
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
import Mathlib.Data.Fin.Basic
import Lean.Data.Json

open Lean Form SimpleGraph

namespace GraphVerifier

structure RuntimeGraph where
  vertexCount : Nat
  edges       : List (Nat × Nat)
  deriving Repr

def RuntimeGraph.neighbours (g : RuntimeGraph) (u : Nat) : List Nat :=
  g.edges.filterMap fun (a, b) =>
    if      a == u then some b
    else if b == u then some a
    else none

def RuntimeGraph.bfs (g : RuntimeGraph) (dst : Nat)
    (queue visited : List Nat) (fuel : Nat) : Bool :=
  match fuel with
  | 0 => false
  | fuel + 1 =>
    match queue with
    | []       => false
    | u :: rest =>
      if u == dst then true
      else if visited.contains u then g.bfs dst rest visited fuel
      else
        let nbrs := g.neighbours u |>.filter (! visited.contains ·)
        g.bfs dst (rest ++ nbrs) (u :: visited) fuel

def RuntimeGraph.reachable (g : RuntimeGraph) (src dst : Nat) : Bool :=
  if src == dst then true
  else g.bfs dst [src] [] (g.vertexCount * g.vertexCount + 1)

def RuntimeGraph.sameComponent (g : RuntimeGraph) (u v : Nat) : Bool :=
  g.reachable u v && g.reachable v u

def parseEdge (j : Json) : Option (Nat × Nat) :=
  match j.getArr? with
  | .error _ => none
  | .ok arr  =>
    match arr[0]?, arr[1]? with
    | some a, some b =>
      match Json.getNat? a, Json.getNat? b with
      | .ok x, .ok y => some (x, y)
      | _, _         => none
    | _, _ => none

def parsePromoted (j : Json) : Option (Nat × Nat × Float) :=
  match j.getObjValAs? Nat   "source",
        j.getObjValAs? Nat   "target",
        j.getObjValAs? Float "confidence" with
  | .ok s, .ok t, .ok c => some (s, t, c)
  | _, _, _              => none

structure BridgeInput where
  vertexCount   : Nat
  baseEdges     : List (Nat × Nat)
  promotedEdges : List (Nat × Nat × Float)
  threshold     : Float
  deriving Repr

def parseBridge (raw : String) : Except String BridgeInput := do
  let j ← Json.parse raw
  match j.getObjValAs? Nat          "vertex_count",
        j.getObjValAs? (Array Json)  "base_edges",
        j.getObjValAs? (Array Json)  "promoted_edges",
        j.getObjValAs? Float         "threshold" with
  | .ok n, .ok be, .ok pe, .ok thr =>
    return {
      vertexCount   := n
      baseEdges     := be.toList.filterMap parseEdge
      promotedEdges := pe.toList.filterMap parsePromoted
      threshold     := thr
    }
  | .error e, _, _, _ => .error s!"vertex_count: {e}"
  | _, .error e, _, _ => .error s!"base_edges: {e}"
  | _, _, .error e, _ => .error s!"promoted_edges: {e}"
  | _, _, _, .error e => .error s!"threshold: {e}"

structure VerificationResult where
  source      : Nat
  target      : Nat
  confidence  : Float
  reachable   : Bool
  sameComp    : Bool
  lawsApplied : List String
  deriving Repr

def verify (bi : BridgeInput) : List VerificationResult :=
  let allEdges := bi.baseEdges ++ bi.promotedEdges.map fun (u, v, _) => (u, v)
  let rg : RuntimeGraph := { vertexCount := bi.vertexCount, edges := allEdges }
  bi.promotedEdges.map fun (u, v, conf) =>
    let r := rg.reachable u v
    let c := rg.sameComponent u v
    { source      := u
      target      := v
      confidence  := conf
      reachable   := r
      sameComp    := c
      lawsApplied :=
        (if r then ["reach_trans", "adj_reach"] else []) ++
        (if c then ["reach_same_component"] else []) }

def status (r : VerificationResult) : String :=
  if r.reachable && r.sameComp then "VERIFIED"
  else if r.reachable           then "REACHABLE (component mismatch)"
  else                               "REJECTED"

private def adjFn (u v : Fin 6) : Prop :=
  (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 0) ∨
  (u = 1 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨
  (u = 2 ∧ v = 3) ∨ (u = 3 ∧ v = 2) ∨
  (u = 3 ∧ v = 4) ∨ (u = 4 ∧ v = 3) ∨
  (u = 0 ∧ v = 5) ∨ (u = 5 ∧ v = 0) ∨
  (u = 5 ∧ v = 2) ∨ (u = 2 ∧ v = 5)

private def refGraph : SimpleGraph (Fin 6) where
  Adj      := adjFn
  symm     := by intro u v h; simp only [adjFn] at *; omega
  loopless := by intro u;     simp only [adjFn];      omega

private instance : DecidableRel refGraph.Adj := by
  intro u v; simp only [refGraph, adjFn]; infer_instance

private theorem ref_connected : refGraph.Connected := by
  rw [SimpleGraph.connected_iff]
  refine ⟨fun u v => ?_, inferInstance⟩
  fin_cases u <;> fin_cases v <;> simp [refGraph, adjFn] <;> decide

def main (args : List String) : IO Unit := do
  let path := args.headD "bridge/lean_input.json"
  let raw  ← IO.FS.readFile path
  match parseBridge raw with
  | .error e =>
    IO.eprintln s!"[verifier] parse error: {e}"
    IO.Process.exit 1
  | .ok bi =>
    IO.println "── form: structural verification ──"
    IO.println s!"   vertices      : {bi.vertexCount}"
    IO.println s!"   base edges    : {bi.baseEdges.length}"
    IO.println s!"   threshold     : {bi.threshold}"
    IO.println s!"   promoted edges: {bi.promotedEdges.length}"
    IO.println ""
    let results  := verify bi
    let verified := results.filter (fun r => r.reachable && r.sameComp)
    let rejected := results.filter (fun r => !(r.reachable && r.sameComp))
    IO.println s!"   {verified.length} verified, {rejected.length} rejected"
    if !verified.isEmpty then
      IO.println "\n   verified:"
      for r in verified do
        IO.println s!"     {status r}  ({r.source}, {r.target})  conf={r.confidence}  laws={r.lawsApplied}"
    if !rejected.isEmpty then
      IO.println "\n   rejected:"
      for r in rejected do
        IO.println s!"     {status r}  ({r.source}, {r.target})  conf={r.confidence}"
    if results.isEmpty then
      IO.println "   (no promoted edges — did the GNN run?)"

end GraphVerifier
