import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

BRIDGE_PATH = "../bridge/facts_and_rules.json"


class NeuroSymbolicGNN(torch.nn.Module):
    """
    GNN encoder (3x GCNConv) + edge MLP for link prediction.

    encode()      → node embeddings capturing structural neighbourhood context
    score_edges() → raw logits for a batch of (src, dst) pairs
    edge_score()  → sigmoid probability for a single (u, v) pair
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
        """edge_pairs: (2, E) — returns raw logits of shape (E,)."""
        src = emb[edge_pairs[0]]
        dst = emb[edge_pairs[1]]
        return self.edge_mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)

    def edge_score(self, emb: torch.Tensor, u: int, v: int) -> float:
        """Sigmoid probability for a single (u, v) pair."""
        src = emb[u].unsqueeze(0)
        dst = emb[v].unsqueeze(0)
        logit = self.edge_mlp(torch.cat([src, dst], dim=-1))
        return torch.sigmoid(logit).item()


def load_graph_from_json(path: str) -> tuple[Data, dict]:
    with open(path) as f:
        raw = json.load(f)
    g = raw["graph"]
    x  = torch.tensor(g["node_features"], dtype=torch.float)
    ei = torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
    # SimpleGraph is undirected — keep both directions
    ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
    y = torch.tensor(g["labels"], dtype=torch.long)
    return Data(x=x, edge_index=ei_sym, y=y), raw


def candidate_edges(n: int) -> list[tuple[int, int]]:
    """All upper-triangle pairs — edges not yet in the graph are candidates."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def train(
    model: NeuroSymbolicGNN,
    data: Data,
    epochs: int = 400,
    lr: float = 5e-3,
    neg_ratio: int = 1,
) -> list[float]:
    """
    Link prediction training with binary cross-entropy.

    neg_ratio=1 keeps class balance on this small graph; higher values
    push the model to be more conservative (higher precision, lower recall).
    lr raised to 5e-3 and epochs to 400 so a 6-node graph actually converges
    within a single run — with 1e-3/300 the model often finishes at loss ~0.69
    (i.e. essentially random), which is why gnn_predictions was always empty.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_ei = data.edge_index
    n = data.x.size(0)
    history: list[float] = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        emb = model.encode(data.x, data.edge_index)

        pos_scores = model.score_edges(emb, pos_ei)
        pos_labels = torch.ones(pos_scores.size(0))

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
        if epoch % 100 == 0:
            print(f"epoch {epoch:>3}  loss={loss.item():.4f}")

    return history


@torch.no_grad()
def predict(
    model: NeuroSymbolicGNN,
    data: Data,
    threshold: float = 0.7,
) -> list[dict]:
    """
    Score all candidate (u, v) pairs; return those above the threshold.

    The threshold is the ontological commitment boundary: predictions above
    it are forwarded as symbolic candidates for Lean verification. The GNN
    does not filter here — the bridge processor applies threshold logic so
    the full score distribution is available for inspection.
    """
    model.eval()
    emb = model.encode(data.x, data.edge_index)
    n = data.x.size(0)

    predictions = []
    for u, v in candidate_edges(n):
        score = model.edge_score(emb, u, v)
        predictions.append({
            "source": u,
            "target": v,
            "confidence": round(score, 4),
            "above_threshold": score >= threshold,
        })

    return predictions


def run_gnn(threshold: float = 0.7):
    dataset, raw = load_graph_from_json(BRIDGE_PATH)

    model = NeuroSymbolicGNN(in_channels=3, hidden_channels=32, out_channels=16)

    print("=== training (link prediction) ===")
    train(model, dataset)

    print("\n=== inference ===")
    all_predictions = predict(model, dataset, threshold=threshold)
    high_conf = [p for p in all_predictions if p["above_threshold"]]

    raw["gnn_predictions"] = all_predictions

    with open(BRIDGE_PATH, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n{len(high_conf)} high-confidence edges (>= {threshold}) written to bridge")
    for p in high_conf:
        print(f"  ({p['source']}, {p['target']})  score={p['confidence']}")

    if not high_conf:
        print("  (none above threshold — try lowering threshold or check training convergence)")


if __name__ == "__main__":
    run_gnn()
