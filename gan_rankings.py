def get_gan_pred_adj_matrix(dataset_name, trial_num):
    from models.gan.utils import read_embeddings
    import numpy as np
    import os

    if dataset_name == "Cora":
        nodes = 2708
        embeddings = 1433
    elif dataset_name == "CiteSeer":
        nodes = 3327
        embeddings = 3703
    elif dataset_name == "PubMed":
        nodes = 19717
        embeddings = 500
    gan_path = f"GAN/results/link_prediction/{dataset_name}/{dataset_name}_trial_{trial_num}_"

    generator = read_embeddings(gan_path + "gen_.emb", nodes, embeddings)
    discriminator = read_embeddings(gan_path + "dis_.emb", nodes, embeddings)
    gen_adj_mat = np.matmul(generator, generator.T)
    dis_adj_mat = np.matmul(discriminator, discriminator.T)
    os.makedirs(f"gan_outputs/{dataset_name}", exist_ok=True)
    np.save(f"gan_outputs/{dataset_name}/trial_{trial_num}_gen_adj_mat.npy", gen_adj_mat)
    np.save(f"gan_outputs/{dataset_name}/trial_{trial_num}_dis_adj_mat.npy", gen_adj_mat)
    return gen_adj_mat, dis_adj_mat

def _read_embeddings(file_path, nodes, n_embeddings):
    """
    Read the embeddings from the file and return them as a numpy array.
    """
    import numpy as np

    with open(file_path, 'r') as f:
        lines = f.readlines()
        embeddings = np.zeros((nodes, n_embeddings))
        for i in range(nodes):
            line = lines[i + 1].strip().split()
            for j in range(n_embeddings):
                embeddings[i][j] = float(line[j + 1])
    return embeddings

def get_gan_rankings(dataset_name, trial_num):
    import numpy as np
    import torch
    import torch_geometric as tg
    from torch_geometric.utils import to_dense_adj
    import pandas as pd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_name == "Cora":
        nodes = 2708
        embeddings = 1433
    elif dataset_name == "CiteSeer":
        nodes = 3327
        embeddings = 3703
    elif dataset_name == "PubMed":
        nodes = 19717
        embeddings = 500
    
    gen_emb = _read_embeddings(f"GAN/results/link_prediction/{dataset_name}/{dataset_name}_trial_{trial_num}_gen_.emb", nodes, embeddings)
    dis_emb = _read_embeddings(f"GAN/results/link_prediction/{dataset_name}/{dataset_name}_trial_{trial_num}_dis_.emb", nodes, embeddings)

    gen_adj_mat = np.load(f"gan_outputs/{dataset_name}/trial_{trial_num}_gen_adj_mat.npy")
    dis_adj_mat = np.load(f"gan_outputs/{dataset_name}/trial_{trial_num}_dis_adj_mat.npy")
    anomalous_graph = torch.load(f"perturbed_data/{dataset_name}/trial_{trial_num}/perturbed_data.pt", weights_only=False, map_location=device)
    original_adj = to_dense_adj(anomalous_graph.edge_index).squeeze(0).cpu().numpy()
    att_anomalous_nodes = torch.load(f"perturbed_data/{dataset_name}/trial_{trial_num}/att_anomalies.pt", weights_only=False, map_location=device).cpu().numpy()
    struct_anomalous_nodes = torch.load(f"perturbed_data/{dataset_name}/trial_{trial_num}/struct_anomalies.pt", weights_only=False, map_location=device).cpu().numpy()
    all_anomalous_nodes = np.concatenate((att_anomalous_nodes, struct_anomalous_nodes), axis=0)
    all_anomalous_nodes = np.unique(all_anomalous_nodes)

    def reconstruction_score(adj_matrix, original_adj, all_anomalous_nodes):
        score = np.sum(np.abs(adj_matrix - original_adj), axis=1)
        sorted_indices = np.argsort(score)[::-1]
        return sorted_indices
    
    def mean_diff_score(embeddings):
        z = np.mean(embeddings,axis=0)
        score = np.linalg.norm(embeddings - z, axis=1)
        sorted_indices = np.argsort(score)
        return sorted_indices

    gen_reconstruction_score = reconstruction_score(gen_adj_mat, original_adj, all_anomalous_nodes)
    dis_reconstruction_score = reconstruction_score(dis_adj_mat, original_adj, all_anomalous_nodes)

    gen_mean_diff_score = mean_diff_score(gen_emb)
    dis_mean_diff_score = mean_diff_score(dis_emb)


    def get_score(anomaly_ranking, anomalies):
        true_one_hot_repr = torch.zeros(nodes)
        true_one_hot_repr[anomalies] = 1
        true_one_hot_repr = true_one_hot_repr.cpu().numpy()
        remapping = [true_one_hot_repr[anomaly_ranking[i]] for i in range(anomalous_graph.x.size(0))]

        inversions = 0
        ones_seen = 0

        for label in remapping:
            if label == 1:
                ones_seen += 1
            else:
                inversions += ones_seen

        return inversions/(ones_seen*(anomalous_graph.x.size(0)-ones_seen))
    
    generator = {"Reconstruction All": [get_score(gen_reconstruction_score, all_anomalous_nodes)],
                 "Reconstruction Structural": [get_score(gen_reconstruction_score, struct_anomalous_nodes)],
                 "Reconstruction Attribute": [get_score(gen_reconstruction_score, att_anomalous_nodes)],
                 "Mean Diff All": [get_score(gen_mean_diff_score, all_anomalous_nodes)],
                 "Mean Diff Structural": [get_score(gen_mean_diff_score, struct_anomalous_nodes)],
                 "Mean Diff Attribute": [get_score(gen_mean_diff_score, att_anomalous_nodes)],
                 "Mean Diff All Reverse": [get_score(gen_mean_diff_score[::-1], all_anomalous_nodes)],
                 "Mean Diff Structural Reverse":[ get_score(gen_mean_diff_score[::-1], struct_anomalous_nodes)],
                 "Mean Diff Attribute Reverse":[ get_score(gen_mean_diff_score[::-1], att_anomalous_nodes)]}
    
    discriminator = {"Reconstruction All": [get_score(dis_reconstruction_score, all_anomalous_nodes)],
                    "Reconstruction Structural": [get_score(dis_reconstruction_score, struct_anomalous_nodes)],
                    "Reconstruction Attribute": [get_score(dis_reconstruction_score, att_anomalous_nodes)],
                    "Mean Diff All": [get_score(dis_mean_diff_score, all_anomalous_nodes)],
                    "Mean Diff Structural": [get_score(dis_mean_diff_score, struct_anomalous_nodes)],
                    "Mean Diff Attribute": [get_score(dis_mean_diff_score, att_anomalous_nodes)],
                    "Mean Diff All Reverse": [get_score(dis_mean_diff_score[::-1], all_anomalous_nodes)],
                    "Mean Diff Structural Reverse": [get_score(dis_mean_diff_score[::-1], struct_anomalous_nodes)],
                    "Mean Diff Attribute Reverse": [get_score(dis_mean_diff_score[::-1], att_anomalous_nodes)]}
    
    discriminator_df = pd.DataFrame(discriminator).T
    generator_df = pd.DataFrame(generator).T
    generator_df.columns = ["Score"]
    discriminator_df.columns = ["Score"]
    generator_df.index = ["Reconstruction Score All", "Reconstruction Score Structural", "Reconstruction Score Attribute",
                            "Mean Diff Score All", "Mean Diff Score Structural", "Mean Diff Score Attribute",
                            "Mean Diff Score All Reverse", "Mean Diff Score Structural Reverse", "Mean Diff Score Attribute Reverse"]
    discriminator_df.index = ["Reconstruction Score All", "Reconstruction Score Structural", "Reconstruction Score Attribute",
                            "Mean Diff Score All", "Mean Diff Score Structural", "Mean Diff Score Attribute",
                            "Mean Diff Score All Reverse", "Mean Diff Score Structural Reverse", "Mean Diff Score Attribute Reverse"]
    joined = pd.concat([generator_df, discriminator_df], axis=1)
    joined.columns = ["Generator Score", "Discriminator Score"]
    joined.to_csv(f"gan_outputs/{dataset_name}/trial_{trial_num}_inversion_scores.csv", index=True)
    print(discriminator_df)
    print()
    print(generator_df)
    exit()

    precision = {"K": [],
                "Generator Reconstruction Score All": [], "Discriminator Reconstruction Score All": [],
                "Generator Mean Diff Score All": [], "Discriminator Mean Diff Score All": [],
                "Generator Reconstruction Score Structural": [], "Discriminator Reconstruction Score Structural": [],
                "Generator Mean Diff Score Structural": [], "Discriminator Mean Diff Score Structural": [],
                "Generator Reconstruction Score Attribute": [], "Discriminator Reconstruction Score Attribute": [],
                "Generator Mean Diff Score Attribute": [], "Discriminator Mean Diff Score Attribute": []}
               
    
    recall = {"K": [], 
              "Generator Reconstruction Score All": [], "Discriminator Reconstruction Score All": [],
                "Generator Mean Diff Score All": [], "Discriminator Mean Diff Score All": [],
                "Generator Reconstruction Score Structural": [], "Discriminator Reconstruction Score Structural": [],
                "Generator Mean Diff Score Structural": [], "Discriminator Mean Diff Score Structural": [],
                "Generator Reconstruction Score Attribute": [], "Discriminator Reconstruction Score Attribute": [],
                "Generator Mean Diff Score Attribute": [], "Discriminator Mean Diff Score Attribute": []}
    
    for i in range(25, len(all_anomalous_nodes) + 1, 25):
        precision["K"].append(i)
        recall["K"].append(i)
        precision["Generator Reconstruction Score All"].append(len(set(gen_reconstruction_score[:i]) & set(all_anomalous_nodes)) / i)
        recall["Generator Reconstruction Score All"].append(len(set(gen_reconstruction_score[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))
        precision["Discriminator Reconstruction Score All"].append(len(set(dis_reconstruction_score[:i]) & set(all_anomalous_nodes)) / i)
        recall["Discriminator Reconstruction Score All"].append(len(set(dis_reconstruction_score[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))
        precision["Generator Mean Diff Score All"].append(len(set(gen_mean_diff_score[:i]) & set(all_anomalous_nodes)) / i)
        recall["Generator Mean Diff Score All"].append(len(set(gen_mean_diff_score[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))
        precision["Discriminator Mean Diff Score All"].append(len(set(dis_mean_diff_score[:i]) & set(all_anomalous_nodes)) / i)
        recall["Discriminator Mean Diff Score All"].append(len(set(dis_mean_diff_score[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))

        precision["Generator Reconstruction Score Structural"].append(len(set(gen_reconstruction_score[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Generator Reconstruction Score Structural"].append(len(set(gen_reconstruction_score[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))
        precision["Discriminator Reconstruction Score Structural"].append(len(set(dis_reconstruction_score[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Discriminator Reconstruction Score Structural"].append(len(set(dis_reconstruction_score[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))
        precision["Generator Mean Diff Score Structural"].append(len(set(gen_mean_diff_score[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Generator Mean Diff Score Structural"].append(len(set(gen_mean_diff_score[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))
        precision["Discriminator Mean Diff Score Structural"].append(len(set(dis_mean_diff_score[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Discriminator Mean Diff Score Structural"].append(len(set(dis_mean_diff_score[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))

        precision["Generator Reconstruction Score Attribute"].append(len(set(gen_reconstruction_score[:i]) & set(att_anomalous_nodes)) / i)
        recall["Generator Reconstruction Score Attribute"].append(len(set(gen_reconstruction_score[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))
        precision["Discriminator Reconstruction Score Attribute"].append(len(set(dis_reconstruction_score[:i]) & set(att_anomalous_nodes)) / i)
        recall["Discriminator Reconstruction Score Attribute"].append(len(set(dis_reconstruction_score[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))
        precision["Generator Mean Diff Score Attribute"].append(len(set(gen_mean_diff_score[:i]) & set(att_anomalous_nodes)) / i)
        recall["Generator Mean Diff Score Attribute"].append(len(set(gen_mean_diff_score[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))
        precision["Discriminator Mean Diff Score Attribute"].append(len(set(dis_mean_diff_score[:i]) & set(att_anomalous_nodes)) / i)
        recall["Discriminator Mean Diff Score Attribute"].append(len(set(dis_mean_diff_score[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))



    precision_df = pd.DataFrame(precision)
    recall_df = pd.DataFrame(recall)
    precision_df.to_csv(f"gan_outputs/{dataset_name}/trial_{trial_num}_gan_precision.csv", index=False)
    recall_df.to_csv(f"gan_outputs/{dataset_name}/trial_{trial_num}_gan_recall.csv", index=False)


def avg_csv_files(dataset_name):
    import pandas as pd
    import os
    print(pd.read_csv(f"gan_outputs/{dataset_name}/trial_{1}_inversion_scores.csv", index_col=0).to_latex())
    exit()
    joined = None
    for i in range(1,4):
        if joined is None:
            joined = pd.read_csv(f"gan_outputs/{dataset_name}/trial_{i}_inversion_scores.csv", index_col=0)
        else:
            temp = pd.read_csv(f"gan_outputs/{dataset_name}/trial_{i}_inversion_scores.csv", index_col=0)
            joined = pd.concat([joined, temp], axis=0)
    print(joined.groupby(joined.index).mean().to_latex())
    joined.to_csv(f"gan_outputs/{dataset_name}/avg_inversion_scores.csv", index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GAN Adjacency Matrix Generation')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--trial_num', type=int, default=0, help='Trial number')
    args = parser.parse_args()

    # gen_adj_mat, dis_adj_mat = get_gan_pred_adj_matrix(args.dataset, args.trial_num)

    #get_gan_rankings(args.dataset, args.trial_num)
    avg_csv_files(args.dataset)