from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data', name='cora')
g = dataset[0]

print(dataset.num_classes)
print(g)
print(g.keys)
print(g.num_features)
print(g.num_node_features)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(g.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.relu = nn.ReLU()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = GCN().to(device)
g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def train(graph):
    model.train()

    out = model(g)
    loss = criterion(out[g.train_mask], g.y[g.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pred = out.argmax(1)
    correct = (pred[g.train_mask] == g.y[g.train_mask]).sum().item()
    train_acc = correct / (g.train_mask.sum().item())

    if i % 10 == 0:
        print("Epoch {:05d}   /   Loss {:.4f}   /   Train_acc {:4f}".
              format(i, loss.item(), train_acc))

def test(graph):
    model.eval()

    out = model(g)
    pred = out.argmax(1)
    correct = (pred[g.test_mask] == g.y[g.test_mask]).sum().item()
    test_acc = correct / (g.test_mask.sum().item())
    print("Test Accuracy : {:.4f}".format(test_acc))

epoch = 200
for i in range(epoch):
    train(g)

test(g)