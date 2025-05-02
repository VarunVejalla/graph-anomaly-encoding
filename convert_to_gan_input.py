"""
This script converts a PyTorch Geometric graph to a format suitable for GAN training.
It generates train, test, and negative test files, these will need to be moved into 
the GAN directory. 
"""

import torch
import torch_geometric.utils as tgu
import torch_geometric.data as tgdata
import torch_geometric.transforms as tgtrans
import os

GAN_DATA_PATH = "GAN"

def convert_pygraph_to_gan_format(data: tgdata.Data, 
                                  data_name: str,
                                  trial_num: int,
                                  test_size: float = 0.1):
    """
    Convert a PyTorch Geometric graph to a format suitable for GAN training.
    """
    
    dataset_folder = f"link_prediction/Cora"
    os.makedirs(GAN_DATA_PATH + "/data/" + dataset_folder, exist_ok=True)
    os.makedirs(GAN_DATA_PATH + "/pre_train/" + dataset_folder, exist_ok=True)
    os.makedirs(GAN_DATA_PATH + "/results/" + dataset_folder, exist_ok=True)
    os.makedirs(GAN_DATA_PATH + "/cache", exist_ok=True)
    
    file_prefix = f"{dataset_folder}/{data_name}_trial_{trial_num}"

    num_nodes = data.x.shape[0]
    train, _, test = tgtrans.RandomLinkSplit(num_val=0, num_test=test_size, is_undirected=True)(data)
    embedding_dim = data.x.shape[1]

    with open(f"{GAN_DATA_PATH}/data/{file_prefix}_train.txt", mode="w") as f:
        for edge in data.edge_index.T:
            f.write(f"{edge[0]} {edge[1]}\n")
    
    with open(f"{GAN_DATA_PATH}/data/{file_prefix}_test.txt", mode="w") as f:
        for edge in test.edge_index.T:
            f.write(f"{edge[0]} {edge[1]}\n")

    embeddings = data.x.cpu().numpy()
    
    with open(f"{GAN_DATA_PATH}/pre_train/{file_prefix}_pre_train.emb", mode="w") as f:
        f.write(f"{num_nodes} {embedding_dim}\n")
        for i in range(num_nodes):
            f.write(f"{i} ")
            for j in range(embedding_dim):
                f.write(f"{embeddings[i][j]} ")
            f.write("\n")
    
    neg_edges = tgu.negative_sampling(test.edge_index,
                                      num_nodes=num_nodes,
                                      num_neg_samples=len(test.edge_index[0]))
    with open(f"{GAN_DATA_PATH}/data/{file_prefix}_test_neg.txt", mode="w") as f:
        for edge in neg_edges.T:
            f.write(f"{edge[0]} {edge[1]}\n")
    return train, test
    
if __name__ == "__main__":
    import sys
    from torch_geometric.datasets import Planetoid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_NAME = sys.argv[1] if len(sys.argv) > 1 else "cora"
    TRIAL_NUM = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    data = torch.load(f"perturbed_data/{DATA_NAME}/trial_{TRIAL_NUM}/perturbed_data.pt", weights_only=False)
    data = data.to(device)
    
    convert_pygraph_to_gan_format(data, DATA_NAME, TRIAL_NUM)
