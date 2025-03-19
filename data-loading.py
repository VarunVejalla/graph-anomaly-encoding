#%%
import torch
from torch_geometric.datasets import Reddit
from torch_geometric.data import DataLoader
import tensorflow as tf
from torch_geometric.nn.models import VGAE

#%%
dataset = Reddit(root="data/Reddit")

print(type(dataset))
# %%
data = dataset[0]

print(data.num_nodes)
print(data.keys())


# data.edge_index is 2 by num_edges tensor 
# column = [i,j] means there is an edge from node i to node j
# data.y is the labels 
#   - not predicting this label. can add as feature or ignore
# data.x is n by d, where n is number of nodes and d is number of features

# how are we using train, val, test split?

# TODO: encode and decode data
# TODO: make anomaly detector for data (missing/new edges, significantly different features)
#       - sort all nodes, most to least likely to be anomaly?

# TODO: 


# might be helpful: https://github.com/Flawless1202/VGAE_pyG/
# also, the VGAE() class
# https://github.com/DaehanKim/vgae_pytorch
# https://antoniolonga.github.io/Pytorch_geometric_tutorials/posts/post6.html

# GAN: https://github.com/hwwang55/GraphGAN
# https://arxiv.org/abs/1711.08267
# https://medium.com/@_psycoplankton/graphgan-generative-adversarial-networks-for-graphs-ff4584375a81

def rank_anomalous(original_data, reconstructed_data):
    
    # score for a node: something like
    # num new edges (or edge deletions) with that node + || original feature - reconstructed feature||
    # or something
    
    scores = [0]*original_data.num_nodes
    for i in range(original_data.num_nodes):
        scores += norm of original_data[i]-reconstructed_data[i]
    
    del_edges = (original_data.edge_index setminus reconstructed_data.edge_index)
    ins_edges = (reconstructed_data.edge_index setminus original_data.edge_index)
    for edge in del_edges.union(ins_edges):
        scores[edge[0]] += 1
        scores[edge[1]] += 1
    
    indices = list(range(original_data.num_nodes))
    indices.sort(key = lambda i: -scores[i])
    return indices