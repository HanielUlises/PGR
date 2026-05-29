# Provable Graph Reasoning

A hybrid neuro-symbolic system for graph reasoning, combining GNN-based
structure inference with Lean 4 metaprogrammed formal verification.

## What this is

Neural networks generalize well over graph-structured data but their outputs
are opaque — there is no principled way to ask *why* a model believes two nodes
are connected, or to guarantee that a set of predictions is internally consistent.
Formal methods offer the opposite trade-off: machine-checked proofs, explicit
inference rules, but rigid encodings that do not adapt to new structure.

This project sits between those two positions. A graph neural network proposes
relational structure — candidate edges, reachability claims, component membership —
and a symbolic layer written in Lean 4 evaluates those proposals against a
rule set that is loaded dynamically at elaboration time rather than compiled into
theorems. The rules themselves are ontological commitments about the domain:
what it means for two nodes to be reachable, what transitivity requires, what
the GNN's confidence score is licensed to assert.

The propositional layer (via `lean-logic-dsl`) acts as a pre-filter: before
any claim reaches the Lean elaborator, its consistency with the current rule set
is checked by a DPLL solver. Only propositions that survive that check are
forwarded for formal verification.

## Research questions

- Can dynamically loaded inference rules serve as a tractable middle ground
  between hardcoded axioms and fully learned representations?
- What is the right interface between a confidence score and a formal claim?
  At what threshold does a GNN prediction license a symbolic commitment?
- How does propositional pre-filtering affect the rate of verification failures
  downstream, and does it catch inconsistencies that the GNN cannot?

## Components

**GNN (PyTorch Geometric)** — trained for link prediction on graph data.
Produces confidence-scored edge candidates as output.

**Bridge processor (Python)** — translates GNN output into a structured format
consumable by the symbolic layer. Applies threshold filtering and formats
predictions as propositions.

**Propositional pre-filter (Lean 4 / lean-logic-dsl)** — encodes GNN assertions
as propositional formulas and runs DPLL satisfiability checking before
verification. Inconsistent prediction sets are rejected here.

**Symbolic core (Lean 4 + Mathlib)** — verifies surviving predictions against
`SimpleGraph` theorems. Rules are loaded from JSON at elaboration time using
Lean metaprogramming, allowing the rule set to change without recompilation.

## Status

Work in progress. The GNN training objective, the bridge data flow, and the
propositional pre-filtering layer are under active development. The Lean
symbolic core is functional for a fixed example graph; generalization to
runtime-constructed graphs is the next milestone.

## Dependencies

- Lean 4 with Mathlib
- [lean-logic-dsl](https://github.com/HanielUlises/lean-logic-dsl) — propositional logic DSL with DPLL
- PyTorch Geometric
