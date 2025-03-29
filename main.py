import os
import torch_geometric
import torch
import torch.nn as nn
from torch.optim import Adam

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from models.VGAE.model import DeepVGAE
from models.VGAE.config.config import parse_args
from sklearn.metrics import classification_report

def run_VGAE():
    """
    Run the VGAE model"
    """

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
    data = torch.load(f"perturbed_data/{args.dataset}/perturbed_data.pt", weights_only=False).to(device)
    # data = torch.load("../../pubmed_anomalous_graph.pt", weights_only=False).to(device)
    # Get the adjacency matrix of the anomalous graph
    orignal_adjacency = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
    # Get the predicted adjacency matrix (link prediction) of the anomalous graph
    pred_adjacency = model.forward(data.x, data.edge_index)
    # Get the top 296 anomalous nodes (nodes encoded with the highest error)
    anomalous = torch.argsort(torch.sum(torch.abs(pred_adjacency - orignal_adjacency), dim=1), descending=True)[:args.num_anomalous_nodes]
    # Load the ground truth anomalous nodes
    # anomalous_nodes = torch.load("../../cora_anomalous_nodes.pt", weights_only=False)
    att_anomalies = torch.load(f"perturbed_data/{args.dataset}/att_anomalies.pt", weights_only=False).to(device)
    struct_anomalies = torch.load(f"perturbed_data/{args.dataset}/struct_anomalies.pt", weights_only=False).to(device)

    att_true_one_hot_repr = torch.zeros(data.x.size(0))
    att_true_one_hot_repr[att_anomalies] = 1
    struct_true_one_hot_repr = torch.zeros(data.x.size(0))
    struct_true_one_hot_repr[struct_anomalies] = 1

    all_anomalies = torch.cat((att_anomalies, struct_anomalies), dim=0).unique()
    true_one_hot_repr = torch.zeros(data.x.size(0))
    true_one_hot_repr[all_anomalies] = 1

    pred_one_hot_repr = torch.zeros(data.x.size(0))
    pred_one_hot_repr[anomalous] = 1

    # print the classification report
    print(classification_report(struct_true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Structural Anomalous"]))
    print(classification_report(att_true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Attribute Anomalous"]))
    print(classification_report(true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Anomalous"]))
    # This is honestly pretty on par with the results from the paper
    # This usually gets an f1-score of about ~0.2 which is pretty good for a simple model
    # The paper's model gets an f1-score of about ~0.4
    # This paper:
    # https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67

if __name__ == "__main__":
    run_VGAE()