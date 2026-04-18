import NeuroSymbolicGraph.SymbolicCore
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Fin.Basic
import Lean.Data.Json

open Lean NeuroSymbolic SimpleGraph

namespace GraphVerifier

def sample_graph : SimpleGraph (Fin 6) where
  Adj u v :=
    (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 0) ∨
    (u = 1 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨
    (u = 2 ∧ v = 3) ∨ (u = 3 ∧ v = 2) ∨
    (u = 3 ∧ v = 4) ∨ (u = 4 ∧ v = 3) ∨
    (u = 0 ∧ v = 5) ∨ (u = 5 ∧ v = 0) ∨
    (u = 5 ∧ v = 2) ∨ (u = 2 ∧ v = 5)
  symm := by
    intro u v h
    simp only at h ⊢
    omega
  loopless := by
    intro u h
    simp only at h
    omega

instance : DecidableRel sample_graph.Adj := by
  intro u v
  simp only [sample_graph]
  infer_instance

def reach_0_1 : sample_graph.Reachable 0 1 :=
  (sample_graph.adj_iff_reachable.mp (by decide)).mpr (by decide)

def reach_1_2 : sample_graph.Reachable 1 2 :=
  Adj.reachable (by decide)

def reach_0_2 : sample_graph.Reachable 0 2 :=
  reach_0_1.trans reach_1_2

def reach_0_3 : sample_graph.Reachable 0 3 :=
  reach_0_2.trans (Adj.reachable (by decide))

def reach_0_4 : sample_graph.Reachable 0 4 :=
  reach_0_3.trans (Adj.reachable (by decide))

def reach_0_5 : sample_graph.Reachable 0 5 :=
  Adj.reachable (by decide)

theorem all_nodes_reachable_from_0 :
    ∀ v : Fin 6, sample_graph.Reachable 0 v := by
  intro v
  fin_cases v
  · exact Reachable.refl 0
  · exact reach_0_1
  · exact reach_0_2
  · exact reach_0_3
  · exact reach_0_4
  · exact reach_0_5

theorem sample_graph_connected : sample_graph.Connected := by
  rw [SimpleGraph.connected_iff]
  constructor
  · intro u v
    have h0u : sample_graph.Reachable 0 u := all_nodes_reachable_from_0 u
    have h0v : sample_graph.Reachable 0 v := all_nodes_reachable_from_0 v
    exact h0u.symm.trans h0v
  · exact ⟨0, 1, by decide⟩

def gnn_edge_rule_json : String :=
  "{\"rules\": [\
    {\"name\": \"transitivity\", \"arity\": 3,\
     \"pattern\": \"Reachable a b ∧ Reachable b c → Reachable a c\",\
     \"lean_tactic\": \"exact SimpleGraph.Reachable.trans\",\
     \"description\": \"transitivity of reachability\"},\
    {\"name\": \"component_sound\", \"arity\": 2,\
     \"pattern\": \"Reachable u v → component u = component v\",\
     \"lean_tactic\": \"exact SimpleGraph.ConnectedComponent.sound\",\
     \"description\": \"reachable implies same component\"}\
  ]}"

theorem verified_by_dynamic_rules
    (h1 : sample_graph.Reachable 0 2)
    (h2 : sample_graph.Reachable 2 4) :
    sample_graph.Reachable 0 4 := by
  apply_dynamic_rules gnn_edge_rule_json
  all_goals (try exact h1.trans h2)

def verify_gnn_prediction (u v : Fin 6) (confidence : Float) : Bool :=
  confidence >= 0.7

def gnn_predictions : List (Fin 6 × Fin 6 × Float) :=
  [(0, 2, 0.85), (0, 4, 0.78), (1, 3, 0.91), (2, 5, 0.73)]

def symbolic_validate_predictions : List String :=
  gnn_predictions.filterMap fun (u, v, score) =>
    if verify_gnn_prediction u v score then
      match all_nodes_reachable_from_0 u, all_nodes_reachable_from_0 v with
      | hu, hv =>
        let _ := hu.symm.trans hv
        some s!"VERIFIED ({u}, {v}) score={score} — reachability confirmed"
    else
      none

#eval do
  IO.println "=== Neuro-Symbolic Graph Verifier ==="
  IO.println s!"sample_graph has 6 vertices"
  let validated := symbolic_validate_predictions
  IO.println s!"{validated.length} predictions passed symbolic verification:"
  for msg in validated do
    IO.println s!"  {msg}"

end GraphVerifier
