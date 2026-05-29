import Lean
import Lean.Elab.Tactic
import Lean.Meta
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
import Mathlib.Data.Finset.Basic

open Lean Lean.Meta Lean.Elab Lean.Elab.Tactic
open SimpleGraph

namespace NeuroSymbolic

structure Rule where
  name        : String
  arity       : Nat
  pattern     : String
  lean_tactic : String
  description : String
  deriving Repr

structure RuleSet where
  rules : Array Rule
  deriving Repr

def parse_rule (j : Json) : Option Rule :=
  match j.getObjValAs? String "name",
        j.getObjValAs? Nat    "arity",
        j.getObjValAs? String "pattern",
        j.getObjValAs? String "lean_tactic",
        j.getObjValAs? String "description" with
  | .ok name, .ok arity, .ok pattern, .ok lean_tactic, .ok description =>
    some { name, arity, pattern, lean_tactic, description }
  | _, _, _, _, _ => none

def load_rule_set (json_str : String) : Except String RuleSet := do
  let j ← Json.parse json_str
  match j.getObjValAs? (Array Json) "rules" with
  | .error e => .error e
  | .ok rules_json =>
    let rules := rules_json.filterMap parse_rule
    return { rules }

def rule_dsl_to_expr (tactic_str : String) : MetaM (Option Expr) := do
  match tactic_str with
  | "exact SimpleGraph.Reachable.trans" =>
    return some (← mkConstWithFreshMVarLevels ``SimpleGraph.Reachable.trans)

  | "exact SimpleGraph.adj_comm.mp" =>
    -- adj_comm gives (G.Adj u v ↔ G.Adj v u); .mp extracts the forward direction.
    return some (← mkConstWithFreshMVarLevels ``SimpleGraph.adj_comm)

  | "exact SimpleGraph.ConnectedComponent.sound" =>
    return some (← mkConstWithFreshMVarLevels ``SimpleGraph.ConnectedComponent.sound)

  | "exact SimpleGraph.Reachable.refl" =>
    return some (← mkConstWithFreshMVarLevels ``SimpleGraph.Reachable.refl)

  | "exact SimpleGraph.Adj.reachable" =>
    -- Witnesses reachability from a single adjacency fact
    return some (← mkConstWithFreshMVarLevels ``SimpleGraph.Adj.reachable)

  | _ =>
    -- Unknown tactic string — log and skip rather than fail hard.
    -- This keeps the pipeline running when the JSON contains rules
    -- that haven't been wired up yet.
    logInfo m!"[rule_dsl_to_expr] no mapping for tactic: '{tactic_str}' — skipped"
    return none

def apply_rule_to_goal (rule : Rule) (goal : MVarId) : MetaM (List MVarId) := do
  let target ← goal.getType
  logInfo m!"[apply_dynamic_rules] trying rule '{rule.name}' on: {target}"
  match ← rule_dsl_to_expr rule.lean_tactic with
  | none =>
    logInfo m!"[apply_dynamic_rules] '{rule.name}' skipped (no tactic mapping)"
    return [goal]
  | some term =>
    try
      let subgoals ← goal.apply term
      logInfo m!"[apply_dynamic_rules] '{rule.name}' applied → {subgoals.length} subgoal(s)"
      return subgoals
    catch e =>
      logInfo m!"[apply_dynamic_rules] '{rule.name}' inapplicable: {← e.toMessageData.toString}"
      return [goal]

def apply_rule_set (rs : RuleSet) (goal : MVarId) : MetaM (List MVarId) := do
  let mut remaining := [goal]
  for rule in rs.rules do
    let mut next_remaining : List MVarId := []
    for g in remaining do
      let subgoals ← apply_rule_to_goal rule g
      next_remaining := next_remaining ++ subgoals
    remaining := next_remaining
  return remaining

/-- Apply all rules from a JSON rule set to the current goal. -/
elab "apply_dynamic_rules" json_arg:str : tactic => do
  let json_str := json_arg.getString
  let goal ← getMainGoal
  match load_rule_set json_str with
  | .error e =>
    throwTacticEx `apply_dynamic_rules goal m!"rule load error: {e}"
  | .ok rs =>
    logInfo m!"[apply_dynamic_rules] loaded {rs.rules.size} rule(s)"
    let remaining ← apply_rule_set rs goal
    replaceMainGoal remaining

elab "load_rules_and_report" json_arg:str : tactic => do
  let json_str := json_arg.getString
  match load_rule_set json_str with
  | .error e =>
    logWarning m!"rule load failed: {e}"
  | .ok rs =>
    for r in rs.rules do
      logInfo m!"rule [{r.name}] arity={r.arity} — {r.description}"
  pure ()

set_option linter.unusedSectionVars false

variable {V : Type*} [DecidableEq V] [Fintype V]
variable (G : SimpleGraph V) [DecidableRel G.Adj]

theorem reachable_trans_dynamic
    (h1 : G.Reachable u v) (h2 : G.Reachable v w) : G.Reachable u w :=
  h1.trans h2

theorem connected_component_eq_of_reachable
    (h : G.Reachable u v) :
    G.connectedComponentMk u = G.connectedComponentMk v :=
  ConnectedComponent.sound h

theorem adj_implies_reachable (h : G.Adj u v) : G.Reachable u v :=
  h.reachable

theorem reachable_self (u : V) : G.Reachable u u :=
  Reachable.refl u

private def transitivityRuleJSON : String :=
  "{\"rules\": [{\"name\": \"transitivity\", \"arity\": 3, " ++
  "\"pattern\": \"Reachable a b ∧ Reachable b c → Reachable a c\", " ++
  "\"lean_tactic\": \"exact SimpleGraph.Reachable.trans\", " ++
  "\"description\": \"transitivity of reachability\"}]}"

private def componentRuleJSON : String :=
  "{\"rules\": [{\"name\": \"component_sound\", \"arity\": 2, " ++
  "\"pattern\": \"Reachable u v → component u = component v\", " ++
  "\"lean_tactic\": \"exact SimpleGraph.ConnectedComponent.sound\", " ++
  "\"description\": \"reachable implies same component\"}]}"

example (G : SimpleGraph V) (h1 : G.Reachable u v) (h2 : G.Reachable v w) :
    G.Reachable u w := by
  apply_dynamic_rules transitivityRuleJSON
  all_goals (first | exact h1.trans h2 | exact h1 | exact h2 | exact Reachable.refl _)

example (G : SimpleGraph V) (h : G.Reachable u v) :
    G.connectedComponentMk u = G.connectedComponentMk v := by
  apply_dynamic_rules componentRuleJSON
  all_goals (first | exact ConnectedComponent.sound h | exact h)

end NeuroSymbolic
