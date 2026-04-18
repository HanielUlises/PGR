# graph-neuro-symbolic

A hybrid neuro-symbolic reasoning pipeline for graph-structured data.
The system combines a graph neural network (GNN) for statistical edge
prediction with a Lean 4 symbolic core that performs machine-checked
verification of inferred structure using dynamically loaded rules.

## Motivation

GNNs generalize well but produce unverifiable outputs. Formal methods
are sound but require manual problem encoding. This project treats them
as complementary: the GNN proposes structure, the symbolic layer
verifies it against a rule set that is loaded at elaboration time, not
hardcoded into theorems.
