import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

BRIDGE_PATH = "../bridge/facts_and_rules.json"

class NeuroSymbolicGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        node_emb = self.conv3(x, edge_index)
        return node_emb

    def edge_score(self, node_emb, u, v):
        pair = torch.cat([node_emb[u], node_emb[v]], dim=-1)
        return self.edge_mlp(pair).item()

def load_graph_from_json(path):
    with open(path) as f:
        data = json.load(f)
    g = data["graph"]
    x = torch.tensor(g["node_features"], dtype=torch.float)
    ei = torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
    ei_sym = torch.cat([ei, ei.flip(0)], dim=1)
    y = torch.tensor(g["labels"], dtype=torch.long)
    return Data(x=x, edge_index=ei_sym, y=y), data

def candidate_edges(n):
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def run_gnn():
    dataset, raw = load_graph_from_json(BRIDGE_PATH)
    n = dataset.x.size(0)

    model = NeuroSymbolicGNN(in_channels=3, hidden_channels=32, out_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        emb = model(dataset.x, dataset.edge_index)
        loss = F.mse_loss(emb.mean(dim=1), dataset.y.float())
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:>3}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        emb = model(dataset.x, dataset.edge_index)

    predictions = []
    threshold = 0.7
    for u, v in candidate_edges(n):
        score = model.edge_score(emb, u, v)
        if score >= threshold:
            predictions.append({
                "source": u,
                "target": v,
                "confidence": round(score, 4),
                "above_threshold": True
            })

    raw["gnn_predictions"] = predictions
    with open(BRIDGE_PATH, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n{len(predictions)} high-confidence edges written to bridge")
    for p in predictions:
        print(f"  ({p['source']}, {p['target']})  score={p['confidence']}")

if __name__ == "__main__":
    run_gnn()