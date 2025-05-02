import os
import torch_geometric
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model import DeepVGAE
from config.config import parse_args
from sklearn.metrics import classification_report

import pickle

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parse_args()

def get_trained_model(args):
    model = DeepVGAE(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    data = torch.load(f"perturbed_data/{args.dataset}/trial_{args.trial_num}/perturbed_data.pt", weights_only=False).to(device)

    all_edge_index = data.edge_index

    # no test set needed
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data.x, all_edge_index, all_edge_index)
        loss.backward()
        optimizer.step()
    
    return model

def get_ranking_reconstruction(data, model):
    original_adjacency = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
    pred_adjacency = model.forward(data.x, data.edge_index)
    difference = torch.abs(pred_adjacency - original_adjacency)
    return torch.argsort(torch.sum(difference, dim=1))#+torch.sum(difference, dim=0)-torch.diagonal(difference))

def get_ranking_encoded_mean_diff(data, model):
    z = model.encode(data.x, data.edge_index)
    z_mean = z.mean(dim=0)
    anomaly_score = torch.norm(z - z_mean, dim=1, p=1)
    return torch.argsort(anomaly_score)

def get_ranking_variance(data, model):
    mu, logvar = model.encoder(data.x, data.edge_index)
    variance = torch.exp(logvar)
    anomaly_score = variance.mean(dim=1)
    print(anomaly_score)
    return torch.argsort(anomaly_score, descending=True)

def get_ranking_gradient_loss(data, model):
    z = model.encode(data.x, data.edge_index)
    z.retain_grad()
    reconstructed = model.decoder.forward_all(z)
    original = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
    loss = F.l1_loss(reconstructed, original, reduction='sum')
    model.zero_grad()
    loss.backward()
    gradients = z.grad
    anomaly_score = gradients.norm(p=2, dim=1)
    return torch.argsort(anomaly_score, descending=True)

def get_ranking_gradient_entropy(data, model):
    z = model.encoder(data.x, data.edge_index)[0]
    z = z.clone().detach().requires_grad_(True)
    adj_pred = model.decoder.forward_all(z)
    adj_true = torch.squeeze(torch_geometric.utils.to_dense_adj(data.edge_index))
    loss = F.binary_cross_entropy(adj_pred, adj_true)
    loss.backward()
    anomaly_score = z.grad.norm(dim=1)
    return torch.argsort(anomaly_score, descending=True)



rankings = {"reconstruction":{},
            "mean_diff":{},
            "variance":{},
            "gradient_loss":{},
            "gradient_entropy":{}
            }

for embedding_dim in list(range(10,401,10)):
    print(embedding_dim)
    args.enc_hidden_channels = []
    args.enc_out_channels = embedding_dim
    model_path = f"models_tmp/{args.dataset}/trial_{args.trial_num}/{args.dataset}-{embedding_dim}-trial_{args.trial_num}.pt"
    
    if os.path.exists(model_path):
        model = DeepVGAE(args).to(device) # loads everything except the weights
        model.load_state_dict(torch.load(model_path))
    
    else:
        model = get_trained_model(args)
        torch.save(model.state_dict(), model_path)
    
    # print(model)
    
    data = torch.load(f"perturbed_data/{args.dataset}/trial_{args.trial_num}/perturbed_data.pt", weights_only=False).to(device)
    
    rankings["reconstruction"][embedding_dim] = get_ranking_reconstruction(data, model)
    rankings["mean_diff"][embedding_dim] = get_ranking_encoded_mean_diff(data, model)
    rankings["variance"][embedding_dim] = get_ranking_variance(data, model)
    rankings["gradient_loss"][embedding_dim] = get_ranking_gradient_loss(data, model)
    rankings["gradient_entropy"][embedding_dim] = get_ranking_gradient_entropy(data, model)
    
    print(max(rankings["reconstruction"][embedding_dim]))
    print(max(rankings["mean_diff"][embedding_dim]))
    print(max(rankings["variance"][embedding_dim]))
    print(max(rankings["gradient_loss"][embedding_dim]))
    print(max(rankings["gradient_entropy"][embedding_dim]))
    
    with open(f"{args.dataset}_ranking_temp_file.pkl", "wb") as file:
        pickle.dump(rankings, file)
    
    
# def get_score(anomaly_ranking):
#     remapping = [true_one_hot_repr[anomaly_ranking[i]] for i in range(data.x.size(0))]

#     inversions = 0
#     ones_seen = 0

#     for label in remapping:
#         if label == 1:
#             ones_seen += 1
#         else:
#             inversions += ones_seen

#     return inversions/(ones_seen*(data.x.size(0)-ones_seen))

# def evaluate(actual_anomalies, anomaly_ranking, num_anomalies):
#     # return precision, recall, F1, and ivnersion score

# anomalous = sorted_loss[-332:]



# Load the ground truth anomalous nodes
# att_anomalies = torch.load(f"perturbed_data/{args.dataset}/att_anomalies.pt", weights_only=False).to(device)
# struct_anomalies = torch.load(f"perturbed_data/{args.dataset}/struct_anomalies.pt", weights_only=False).to(device)

# all_anomalies = torch.cat((att_anomalies, struct_anomalies), dim=0).unique()
# true_one_hot_repr = torch.zeros(data.x.size(0))
# true_one_hot_repr[all_anomalies] = 1


# pred_one_hot_repr = torch.zeros(data.x.size(0))

# anomaly_set = set(all_anomalies)


# for i, ranking in enumerate([anomaly_ranking_1, anomaly_ranking_2, anomaly_ranking_3, anomaly_ranking_4, anomaly_ranking_5]):
#     print(i+1, get_score(ranking))

# att_true_one_hot_repr = torch.zeros(data.x.size(0))
# att_true_one_hot_repr[att_anomalies] = 1
# struct_true_one_hot_repr = torch.zeros(data.x.size(0))
# struct_true_one_hot_repr[struct_anomalies] = 1

# print the classification report
# print(classification_report(struct_true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Structural Anomalous"]))
# print(classification_report(att_true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Attribute Anomalous"]))
# print(classification_report(true_one_hot_repr, pred_one_hot_repr, target_names=["Normal", "Anomalous"]))
# This is honestly pretty on par with the results from the paper
# This usually gets an f1-score of about ~0.2 which is pretty good for a simple model
# The paper's model gets an f1-score of about ~0.4
# This paper:
# https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67
        
