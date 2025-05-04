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
torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())


def get_score(anomaly_ranking, true_one_hot_repr):
    # print(len(anomaly_ranking))
    size = len(anomaly_ranking)
    remapping = [true_one_hot_repr[anomaly_ranking[i]] for i in range(size)]

    inversions = 0
    ones_seen = 0

    for label in remapping:
        if label == 1:
            ones_seen += 1
        else:
            inversions += ones_seen

    return inversions/(ones_seen*(size-ones_seen))

def make_plot(anomaly_type = "all", ds = "all", tn = -1):
    anomaly_type = anomaly_type.lower()
    ds = ds.lower()
    assert anomaly_type in ["all", "attribute", "structural"]
    assert ds in ["all", "citeseer", "cora", "pubmed"]
    assert tn in [-1, 1, 2, 3]
    
    if ds == "all":
        datasets = ["citeseer", "cora", "pubmed"]
    else:
        datasets = [ds]
    if tn == -1:
        trial_nums = [1,2,3]
    else:
        trial_nums = [tn]
    
    plt.xlim(0, 400)
    plt.ylim(0, 1)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Inversion Score')
    
    total_scores = {}
    
    for dataset in datasets:
        for trial_num in trial_nums:
            print(dataset, trial_num)
            data_path = f"perturbed_data/{dataset}/trial_{trial_num}"

            data = torch.load(f"{data_path}/perturbed_data.pt", weights_only=False, map_location=torch.device('cpu')).to(device)

            
            att_anomalies = torch.load(f"{data_path}/att_anomalies.pt", weights_only=False, map_location=torch.device('cpu')).to(device)
            struct_anomalies = torch.load(f"{data_path}/struct_anomalies.pt", weights_only=False, map_location=torch.device('cpu')).to(device)
            
            if anomaly_type.lower() == "all":
                all_anomalies = torch.cat((att_anomalies, struct_anomalies), dim=0).unique()
            elif anomaly_type.lower() == "attribute":
                all_anomalies = torch.cat((att_anomalies, ), dim=0).unique()
            elif anomaly_type.lower() == "structural":
                all_anomalies = torch.cat((struct_anomalies, ), dim=0).unique()
            

            
            true_one_hot_repr = torch.zeros(data.x.size(0))
            true_one_hot_repr[all_anomalies] = 1

            # print(max(set(all_anomalies)), data.x.size(0))

            # def evaluate(actual_anomalies, anomaly_ranking, num_anomalies):
            #     # return precision, recall, F1, and inversion score

            # anomalous = sorted_loss[-332:]

            with open(f"vgae_outputs/{dataset}_ranking_trial_{trial_num}.pkl", "rb") as file:
                rankings = pickle.load(file)

            num_anomalies = len(all_anomalies)

            # print(rankings.keys())

            # pred_one_hot_repr = torch.zeros(data.x.size(0))

            # anomaly_set = set(all_anomalies)

            # scores = {key:[[], []] for key in rankings.keys()}

            for metric, ranking_dict in rankings.items():
                if metric not in total_scores:
                    total_scores[metric] = {}
                for embedding_dim, ranking in ranking_dict.items():
                    if embedding_dim not in total_scores[metric]:
                        total_scores[metric][embedding_dim] = [0, 0]
                    
                    if metric in ["gradient_entropy", "gradient_loss"]:
                        curr_score = get_score(ranking.flip(0), true_one_hot_repr)
                    else:
                        curr_score = get_score(ranking, true_one_hot_repr)
                    
                    total_scores[metric][embedding_dim][0] += curr_score
                    total_scores[metric][embedding_dim][1] += 1
    
    avg_scores = {metric: [[], []] for metric in total_scores.keys()}
    for metric in total_scores:
        for embedding_dim in total_scores[metric]:
            avg_scores[metric][0].append(embedding_dim)
            lu = total_scores[metric][embedding_dim]
            avg_scores[metric][1].append(lu[0]/lu[1])

    for metric, (embedding_dims, metric_scores) in avg_scores.items():
        plt.plot(embedding_dims, metric_scores, label=metric)
    
    if tn == -1:
        plt.title(f'All Datasets - {anomaly_type[0].upper() + anomaly_type[1:]} Anomalies')
    else:
        plt.title(f'All Datasets - {anomaly_type[0].upper() + anomaly_type[1:]} Anomalies, trial {tn}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"images/{ds}_{tn}_{anomaly_type}.png")
    return avg_scores

for anomaly_type in ["attribute", "structural", "all"]:
    for ds in ["all"]:
        for tn in [-1]:
            print(anomaly_type, ds)
            res = make_plot(anomaly_type, ds, tn)
            plt.close()
            # print(anomaly_type, ds)
            # print("reconstruction", sum(res["reconstruction"][1])/len(res["reconstruction"][1]))
            # print("mean_diff", sum(res["mean_diff"][1])/len(res["mean_diff"][1]))
            