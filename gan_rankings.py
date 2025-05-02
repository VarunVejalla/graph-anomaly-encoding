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

def vgae_rankings(dataset_name, trial_num):
    import numpy as np
    import torch
    import pandas as pd
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(f"vgae_outputs/{dataset_name}", exist_ok=True)

    data = np.load(f"models_tmp/{dataset_name}/{dataset_name}_ranking_temp_file_trial_2.pkl", allow_pickle=True)
    reconstruction_ranking = data['reconstruction'][170].cpu().numpy()
    mean_diff_ranking = data['mean_diff'][170].cpu().numpy()
    att_anomalous_nodes = torch.load(f"perturbed_data/{dataset_name}/trial_{trial_num}/att_anomalies.pt", weights_only=False, map_location=device).cpu().numpy()
    struct_anomalous_nodes = torch.load(f"perturbed_data/{dataset_name}/trial_{trial_num}/struct_anomalies.pt", weights_only=False, map_location=device).cpu().numpy()
    all_anomalous_nodes = np.concatenate((att_anomalous_nodes, struct_anomalous_nodes), axis=0)
    all_anomalous_nodes = np.unique(all_anomalous_nodes)
    precision = {"K": [],
                "Reconstruction Score All": [], "Mean Diff Score All": [],
                "Reconstruction Score Structural": [], "Mean Diff Score Structural": [],
                "Reconstruction Score Attribute": [], "Mean Diff Score Attribute": []}
    
    recall = {"K": [],
                "Reconstruction Score All": [], "Mean Diff Score All": [],
                    "Reconstruction Score Structural": [], "Mean Diff Score Structural": [],
                    "Reconstruction Score Attribute": [], "Mean Diff Score Attribute": []}
    
    for i in range(25, len(all_anomalous_nodes) + 1, 25):
        precision["K"].append(i)
        recall["K"].append(i)
        precision["Reconstruction Score All"].append(len(set(reconstruction_ranking[:i]) & set(all_anomalous_nodes)) / i)
        recall["Reconstruction Score All"].append(len(set(reconstruction_ranking[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))
        precision["Mean Diff Score All"].append(len(set(mean_diff_ranking[:i]) & set(all_anomalous_nodes)) / i)
        recall["Mean Diff Score All"].append(len(set(mean_diff_ranking[:i]) & set(all_anomalous_nodes)) / len(all_anomalous_nodes))

        precision["Reconstruction Score Structural"].append(len(set(reconstruction_ranking[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Reconstruction Score Structural"].append(len(set(reconstruction_ranking[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))
        precision["Mean Diff Score Structural"].append(len(set(mean_diff_ranking[:i]) & set(struct_anomalous_nodes)) / i)
        recall["Mean Diff Score Structural"].append(len(set(mean_diff_ranking[:i]) & set(struct_anomalous_nodes)) / len(struct_anomalous_nodes))

        precision["Reconstruction Score Attribute"].append(len(set(reconstruction_ranking[:i]) & set(att_anomalous_nodes)) / i)
        recall["Reconstruction Score Attribute"].append(len(set(reconstruction_ranking[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))
        precision["Mean Diff Score Attribute"].append(len(set(mean_diff_ranking[:i]) & set(att_anomalous_nodes)) / i)
        recall["Mean Diff Score Attribute"].append(len(set(mean_diff_ranking[:i]) & set(att_anomalous_nodes)) / len(att_anomalous_nodes))
    precision_df = pd.DataFrame(precision)
    recall_df = pd.DataFrame(recall)
    precision_df.to_csv(f"vgae_outputs/{dataset_name}/trial_{trial_num}_vgae_precision.csv", index=False)
    recall_df.to_csv(f"vgae_outputs/{dataset_name}/trial_{trial_num}_vgae_recall.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GAN Adjacency Matrix Generation')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--trial_num', type=int, default=0, help='Trial number')
    args = parser.parse_args()

    # gen_adj_mat, dis_adj_mat = get_gan_pred_adj_matrix(args.dataset, args.trial_num)

    # get_gan_rankings(args.dataset, args.trial_num)

    vgae_rankings(args.dataset, args.trial_num)