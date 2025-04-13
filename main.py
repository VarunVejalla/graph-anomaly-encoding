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
    data = torch.load(f"perturbed_data/{args.dataset}/perturbed_data.pt", weights_only=False, map_location=torch.device(device)).to(device)
    # data = torch.load("../../pubmed_anomalous_graph.pt", weights_only=False).to(device)
    # Get the adjacency matrix of the anomalous graph
    orignal_adjacency = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
    # Get the predicted adjacency matrix (link prediction) of the anomalous graph
    pred_adjacency = model.forward(data.x, data.edge_index)
    # Get the top 296 anomalous nodes (nodes encoded with the highest error)
    anomalous = torch.argsort(torch.sum(torch.abs(pred_adjacency - orignal_adjacency), dim=1), descending=True)[:args.num_anomalous_nodes]
    # Load the ground truth anomalous nodes
    # anomalous_nodes = torch.load("../../cora_anomalous_nodes.pt", weights_only=False)
    att_anomalies = torch.load(f"perturbed_data/{args.dataset}/att_anomalies.pt", weights_only=False, map_location=torch.device(device)).to(device)
    struct_anomalies = torch.load(f"perturbed_data/{args.dataset}/struct_anomalies.pt", weights_only=False, map_location=torch.device(device)).to(device)

    att_true_one_hot_repr = torch.zeros(data.x.size(0)).to(device)
    att_true_one_hot_repr[att_anomalies] = 1
    struct_true_one_hot_repr = torch.zeros(data.x.size(0)).to(device)
    struct_true_one_hot_repr[struct_anomalies] = 1

    all_anomalies = torch.cat((att_anomalies, struct_anomalies), dim=0).unique()
    true_one_hot_repr = torch.zeros(data.x.size(0)).to(device)
    true_one_hot_repr[all_anomalies] = 1

    pred_one_hot_repr = torch.zeros(data.x.size(0)).to(device)
    pred_one_hot_repr[anomalous] = 1

    # print the classification report
    true_anomalies = set(all_anomalies.tolist())
    print("PRECISION/RECALL AT K\nK\tPRECISION\tRECALL")
    for k in range(50, args.num_anomalous_nodes + 1, 50):
        preds = anomalous[:k].tolist()
        preds = set(preds)
        # Get the intersection of the two sets
        intersection = preds.intersection(true_anomalies)
        precision = len(intersection) / k
        
        recall = len(intersection) / len(true_anomalies)
        # print the precision and recall
        print(f"{k}\t{precision:.4f}\t{recall:.4f}")

    print()
    print(f"Percent of structural anomalies identified: {torch.sum(pred_one_hot_repr[struct_anomalies])/struct_anomalies.shape[0] * 100:.4f}% ({torch.sum(pred_one_hot_repr[struct_anomalies])} / {struct_anomalies.shape[0]})")
    print(f"Percent of attribute anomalies identified: {torch.sum(pred_one_hot_repr[att_anomalies])/att_anomalies.shape[0] * 100:.4f}% ({torch.sum(pred_one_hot_repr[att_anomalies])} / {att_anomalies.shape[0]})")

    
    # https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67

if __name__ == "__main__":
    run_VGAE()