"""
content/gnn_model.py

The content side of the form/content split.

The GNN is responsible for everything empirical and uncertain:
  - what nodes exist and what they look like (degree-based structural features)
  - which edges are likely                   (link prediction)
  - how confident each prediction is         (sigmoid score in [0, 1])

It knows nothing about structural laws. It produces a scored candidate list
and hands it to the bridge. The form side (Lean) decides what is structurally
admissible.
"""

import json
import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, to_undirected, degree

GRAPH_PATH       = "../bridge/graph.json"
PREDICTIONS_PATH = "../bridge/predictions.json"

def structural_features(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Degree-based node features derived entirely from topology.
    No hand-crafted placeholder values. Works for any graph of any size.

      col 0  raw degree
      col 1  degree / max_degree      (normalised, in [0,1])
      col 2  log(degree + 1)          (compresses high-degree hubs)
    """
    deg     = degree(edge_index[0], num_nodes=num_nodes).float()
    max_deg = deg.max().clamp(min=1.0)
    return torch.stack([deg, deg / max_deg, (deg + 1.0).log()], dim=1)


class ContentGNN(torch.nn.Module):
    """
    3-layer GCN encoder + edge MLP for link prediction.

    Edge representation: [src | dst | src*dst | |src-dst|]
      concat          directional context
      Hadamard        permutation-invariant similarity (undirected-friendly)
      abs-difference  structural dissimilarity
    """

    def __init__(self, in_channels: int, hidden: int = 32, out: int = 16):
        super().__init__()
        self.conv1    = GCNConv(in_channels, hidden)
        self.conv2    = GCNConv(hidden, hidden)
        self.conv3    = GCNConv(hidden, out)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(out * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv3(x, edge_index)

    def score_edges(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        src, dst = emb[pairs[0]], emb[pairs[1]]
        return self.edge_mlp(
            torch.cat([src, dst, src * dst, (src - dst).abs()], dim=-1)
        ).squeeze(-1)

    @torch.no_grad()
    def score_all_pairs(
        self, emb: torch.Tensor
    ) -> tuple[list[tuple[int, int]], torch.Tensor]:
        """Single vectorised pass over every upper-triangle (u, v) pair."""
        n  = emb.size(0)
        us = torch.tensor([u for u in range(n) for v in range(u + 1, n)])
        vs = torch.tensor([v for u in range(n) for v in range(u + 1, n)])
        return list(zip(us.tolist(), vs.tolist())), self.score_edges(emb, torch.stack([us, vs]))

def load_graph(path: str = GRAPH_PATH) -> tuple[Data, dict]:
    with open(path) as f:
        raw = json.load(f)
    g      = raw["graph"]
    ei     = torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
    ei_sym = to_undirected(ei)
    n      = g["vertex_count"]
    x      = structural_features(ei_sym, num_nodes=n)
    y      = torch.tensor(g.get("labels", [0] * n), dtype=torch.long)
    return Data(x=x, edge_index=ei_sym, num_nodes=n, y=y), raw

def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    p = pos[:2000].unsqueeze(1)
    n = neg[:2000].unsqueeze(0)
    return ((p > n).float().mean() + 0.5 * (p == n).float().mean()).item()


def train(
    model: ContentGNN,
    data: Data,
    epochs: int = 400,
    lr: float = 5e-3,
    neg_ratio: int = 1,
    val_fraction: float = 0.1,
) -> list[dict]:
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n       = data.x.size(0)
    num_pos = data.edge_index.size(1)
    num_val = max(1, int(num_pos * val_fraction))
    history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        emb      = model.encode(data.x, data.edge_index)
        perm     = torch.randperm(num_pos)
        train_ei = data.edge_index[:, perm[num_val:]]
        val_ei   = data.edge_index[:, perm[:num_val]]

        pos_s  = model.score_edges(emb, train_ei)
        neg_ei = negative_sampling(data.edge_index, num_nodes=n,
                                   num_neg_samples=train_ei.size(1) * neg_ratio)
        neg_s  = model.score_edges(emb, neg_ei)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_s, neg_s]),
            torch.cat([torch.ones(pos_s.size(0)), torch.zeros(neg_s.size(0))]),
        )
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            ve    = model.encode(data.x, data.edge_index)
            vp    = torch.sigmoid(model.score_edges(ve, val_ei))
            vn_ei = negative_sampling(data.edge_index, num_nodes=n, num_neg_samples=num_val)
            vn    = torch.sigmoid(model.score_edges(ve, vn_ei))
            auc   = _auc(vp, vn)

        history.append({"epoch": epoch, "loss": loss.item(), "val_auc": auc})
        if epoch % 100 == 0:
            print(f"   epoch {epoch:>3}  loss={loss.item():.4f}  val_auc={auc:.3f}")

    return history

@torch.no_grad()
def predict(model: ContentGNN, data: Data) -> list[dict]:
    """
    Score all candidate pairs in one vectorised pass.
    No threshold applied here — the bridge decides what gets promoted to Lean.
    """
    model.eval()
    emb           = model.encode(data.x, data.edge_index)
    pairs, logits = model.score_all_pairs(emb)
    scores        = torch.sigmoid(logits)
    return [
        {"source": u, "target": v, "confidence": round(s, 4)}
        for (u, v), s in zip(pairs, scores.tolist())
    ]

def run(threshold: float = 0.7):
    data, raw = load_graph()
    model     = ContentGNN(in_channels=3)

    print("── content: training ──")
    history  = train(model, data)
    best_auc = max((r["val_auc"] for r in history
                    if not math.isnan(r["val_auc"])), default=float("nan"))
    print(f"   best val_auc = {best_auc:.3f}")

    print("── content: inference ──")
    predictions = predict(model, data)

    out = {
        "vertex_count": data.num_nodes,
        "base_edges":   raw["graph"]["edges"],
        "threshold":    threshold,
        "predictions":  predictions,
    }
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    above = [p for p in predictions if p["confidence"] >= threshold]
    print(f"   {len(above)}/{len(predictions)} pairs above threshold {threshold}")
    for p in above:
        print(f"   ({p['source']}, {p['target']})  {p['confidence']:.4f}")
    if not above:
        print("   (none — lower THRESHOLD or check val_auc)")


if __name__ == "__main__":
    run()
