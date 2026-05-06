import json
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

BRIDGE_PATH = "../bridge/facts_and_rules.json"


class NeuroSymbolicGNN(torch.nn.Module):
    """
    GNN encoder + edge MLP for link prediction.

    The encoder (3x GCNConv) produces node embeddings that capture
    structural neighbourhood context. The edge MLP scores pairs of
    embeddings, returning a probability that the edge exists.

    Training objective: binary cross-entropy on known positive edges
    vs. randomly sampled negative edges (edges not in the graph).
    This is the standard approach for link prediction — the node-label
    MSE in the original code never backpropagated into the edge MLP at all.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv3(x, edge_index)

    def score_edges(self, emb: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        edge_pairs: (2, E) tensor of (source, target) indices.
        Returns raw logits of shape (E,).
        """
        src = emb[edge_pairs[0]]
        dst = emb[edge_pairs[1]]
        return self.edge_mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)

    def edge_score(self, emb: torch.Tensor, u: int, v: int) -> float:
        """Sigmoid probability for a single (u, v) pair."""
        pair = torch.stack([emb[u], emb[v]], dim=0).unsqueeze(0)
        src = emb[u].unsqueeze(0)
        dst = emb[v].unsqueeze(0)
        logit = self.edge_mlp(torch.cat([src, dst], dim=-1))
        return torch.sigmoid(logit).item()


def load_graph_from_json(path: str) -> tuple[Data, dict]:
    with open(path) as f:
        data = json.load(f)
    g = data["graph"]
    x  = torch.tensor(g["node_features"], dtype=torch.float)
    ei = torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
    # SimpleGraph is undirected — keep both directions
    ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
    y = torch.tensor(g["labels"], dtype=torch.long)
    return Data(x=x, edge_index=ei_sym, y=y), data


def candidate_edges(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def train(
    model: NeuroSymbolicGNN,
    data: Data,
    epochs: int = 300,
    lr: float = 1e-3,
    neg_ratio: int = 2,
) -> list[float]:
    """
    Link prediction training loop.

    For each epoch:
      1. Encode all nodes into embeddings.
      2. Score the known positive edges (those in the graph).
      3. Sample `neg_ratio` negative edges per positive edge using
         PyG's negative_sampling (guaranteed not to be in the graph).
      4. Compute binary cross-entropy between positive scores (label=1)
         and negative scores (label=0).
      5. Backpropagate — gradients now flow through both the GCN encoder
         and the edge MLP, which is what was missing before.

    neg_ratio controls class balance. With a sparse graph, negatives
    vastly outnumber positives, so we subsample to avoid trivial solutions
    where the model predicts 0 for everything.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Only unique directed positive edges (upper triangle)
    pos_ei = data.edge_index
    n = data.x.size(0)
    history = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        emb = model.encode(data.x, data.edge_index)

        # Positive edge scores
        pos_scores = model.score_edges(emb, pos_ei)
        pos_labels = torch.ones(pos_scores.size(0))

        # Negative sampling: sample edges not in the graph
        neg_ei = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=n,
            num_neg_samples=pos_ei.size(1) * neg_ratio,
        )
        neg_scores = model.score_edges(emb, neg_ei)
        neg_labels = torch.zeros(neg_scores.size(0))

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % 50 == 0:
            print(f"epoch {epoch:>3}  loss={loss.item():.4f}")

    return history


@torch.no_grad()
def predict(
    model: NeuroSymbolicGNN,
    data: Data,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Run inference over all candidate (u, v) pairs and return those
    whose edge probability exceeds the threshold.

    The threshold is the ontological commitment boundary: predictions
    above it are treated as symbolic candidates for Lean verification.
    Predictions below it are discarded — the GNN is not confident enough
    to warrant a formal claim.
    """
    model.eval()
    emb = model.encode(data.x, data.edge_index)
    n = data.x.size(0)

    predictions = []
    for u, v in candidate_edges(n):
        score = model.edge_score(emb, u, v)
        if score >= threshold:
            predictions.append({
                "source": u,
                "target": v,
                "confidence": round(score, 4),
                "above_threshold": True,
            })

    return predictions


def run_gnn():
    dataset, raw = load_graph_from_json(BRIDGE_PATH)

    model = NeuroSymbolicGNN(in_channels=3, hidden_channels=32, out_channels=16)

    print("=== training (link prediction) ===")
    train(model, dataset, epochs=300)

    print("\n=== inference ===")
    predictions = predict(model, dataset, threshold=0.7)

    raw["gnn_predictions"] = predictions
    with open(BRIDGE_PATH, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n{len(predictions)} high-confidence edges written to bridge")
    for p in predictions:
        print(f"  ({p['source']}, {p['target']})  score={p['confidence']}")


if __name__ == "__main__":
    run_gnn()