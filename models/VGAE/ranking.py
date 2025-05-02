import torch

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model import DeepVGAE
from config.config import parse_args
from sklearn.metrics import classification_report

import pickle

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parse_args()
data_path = f"perturbed_data/{args.dataset}/trial_{args.trial_num}"

data = torch.load(f"{data_path}/perturbed_data.pt", weights_only=False).to(device)

att_anomalies = torch.load(f"{data_path}/att_anomalies.pt", weights_only=False).to(device)
struct_anomalies = torch.load(f"{data_path}/struct_anomalies.pt", weights_only=False).to(device)

all_anomalies = torch.cat((att_anomalies, struct_anomalies), dim=0).unique()
true_one_hot_repr = torch.zeros(data.x.size(0))
true_one_hot_repr[all_anomalies] = 1

print(max(set(all_anomalies)), data.x.size(0))

def get_score(anomaly_ranking):
    # print(len(anomaly_ranking))
    remapping = [true_one_hot_repr[anomaly_ranking[i]] for i in range(data.x.size(0))]

    inversions = 0
    ones_seen = 0

    for label in remapping:
        if label == 1:
            ones_seen += 1
        else:
            inversions += ones_seen

    return inversions/(ones_seen*(data.x.size(0)-ones_seen))

# def evaluate(actual_anomalies, anomaly_ranking, num_anomalies):
#     # return precision, recall, F1, and inversion score

# anomalous = sorted_loss[-332:]

with open(f"{args.dataset}_ranking_temp_file.pkl", "rb") as file:
    rankings = pickle.load(file)

num_anomalies = len(all_anomalies)

print(rankings.keys())

# pred_one_hot_repr = torch.zeros(data.x.size(0))

# anomaly_set = set(all_anomalies)

scores = {key:[[], []] for key in rankings.keys()}

for metric, ranking_dict in rankings.items():
    for embedding_dim, ranking in ranking_dict.items():
        # print(len(ranking))
        scores[metric][0].append(embedding_dim)
        if metric in ["gradient_entropy", "gradient_loss"]:
            scores[metric][1].append(get_score(ranking.flip(0)))
        else:
            scores[metric][1].append(get_score(ranking))

for metric, (embedding_dims, metric_scores) in scores.items():
    plt.plot(embedding_dims, metric_scores, label=metric)

plt.xlim(0, 400)
plt.ylim(0, 1)
plt.xlabel('Embedding Dimension')
plt.ylabel('Inversion Score')
plt.title(f'{args.dataset} - attribute anomalies')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(f"images/{args.dataset}_attribute.png")



# att_true_one_hot_repr = torch.zeros(data.x.size(0))
# att_true_one_hot_repr[att_anomalies] = 1
# struct_true_one_hot_repr = torch.zeros(data.x.size(0))
# struct_true_one_hot_repr[struct_anomalies] = 1