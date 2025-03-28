# help determine what k-cliques we should use based on datasaet
import networkx as nx
from math import comb

edges_file = "cora.edges"
node_labels_file = "cora.node_labels"

G = nx.Graph()
with open(edges_file) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            G.add_edge(u, v)

node_labels = {}
with open(node_labels_file) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) != 2:
            continue
        node, label = int(parts[0]), int(parts[1])
        node_labels[node] = label

cliques = [clique for clique in nx.find_cliques(G) if len(clique) == 4]
num_10_cliques = len(cliques)


n = G.number_of_nodes()

total_possible_10_node_combinations = comb(n, 4)

proportion = num_10_cliques / total_possible_10_node_combinations if total_possible_10_node_combinations else 0

print(f"Number of nodes: {n}")
print(f"Actual number of 10-cliques: {num_10_cliques}")
print(f"Total possible 10-node combinations: {total_possible_10_node_combinations}")
print(f"Proportion of actual 10-cliques: {proportion:.10f}")


