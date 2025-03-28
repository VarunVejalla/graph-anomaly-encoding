import os
import torch_geometric
import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model import DeepVGAE
from config.config import parse_args

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parse_args()

model = DeepVGAE(args).to(device)
optimizer = Adam(model.parameters(), lr=args.lr)

os.makedirs("datasets", exist_ok=True)
dataset = Planetoid("datasets", args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
all_edge_index = data.edge_index
data = train_test_split_edges(data, 0.05, 0.1)

for epoch in range(args.epoch):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        model.eval()
        roc_auc, ap = model.single_test(data.x,
                                        data.train_pos_edge_index,
                                        data.test_pos_edge_index,
                                        data.test_neg_edge_index)
        print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
        
# Load anomalous graph
data = torch.load("../../cora_anomalous_graph.pt", weights_only=False)
# Get the adjacency matrix of the anomalous graph
orignal_adjacency = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
# Get the predicted adjacency matrix (link prediction) of the anomalous graph
pred_adjacency = model.forward(data.x, data.edge_index)
# Get the top 296 anomalous nodes (nodes encoded with the highest error)
anomalous = torch.argsort(torch.sum(torch.abs(pred_adjacency - orignal_adjacency), dim=1), descending=True)[:296]
# Load the ground truth anomalous nodes
anomalous_nodes = torch.load("../../cora_anomalous_nodes.pt", weights_only=False)

# Loop through the top 296 anomalous nodes and compare with the ground truth anomalous nodes
correct = 0
incorrect = 0
for node in anomalous:
    if node in anomalous_nodes:
        correct += 1
    else:
        incorrect += 1
        
