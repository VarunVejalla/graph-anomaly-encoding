from random import randrange, sample
import torch
import torch_geometric.data as tgdata
import torch_geometric.utils as tgutils

def structurally_perturb(graph: tgdata.Data, anomalous_nodes: torch.Tensor, clique_size: int) -> tuple[tgdata.Data, torch.Tensor]:
    """
    Creates cliques of anomalous nodes in the graph.

    Args:
        graph (tgdata.Data): The input graph.
        anomalous_nodes (torch.Tensor): The nodes to create cliques for.
        clique_size (int): The size of the cliques to create.
    
    Returns:
        tuple: A tuple containing the new graph and the anomalous nodes.
    """ 
    
    new_graph = graph.clone()

    # Reshape the anomalous nodes to be (-1, clique_size) the -1 will be the number of cliques and is inferred by pytorch
    nodes_in_cliques = anomalous_nodes.reshape(-1, clique_size) 
    print("Anomalous nodes shape:", nodes_in_cliques.shape)
    # Creates a set of tuples to represent the edges to prevent duplicates
    anomalous_edges = set()
    # Iterate through the cliques
    for clique in nodes_in_cliques:
        # Iterate through the nodes in each clique
        for i, node in enumerate(clique):
            # Create edges between the nodes in the clique
            for j in range(i + 1, len(clique)):
                anomalous_edges.add((node.item(), clique[j].item())) # Pytorch represents undirected edges as (u, v) and (v, u)
                anomalous_edges.add((clique[j].item(), node.item())) # Add the reverse edge as well
    
    # Convert the set of edges back to a tensor
    anomalous_edges = torch.tensor(list(anomalous_edges), device=graph.edge_index.device).T
    new_graph.edge_index = torch.concat([graph.edge_index, anomalous_edges], dim=1)

    # Coalese sorts the edges and removes duplicates prevents having to convert to a list and set and back to a tensor
    new_graph.edge_index, new_graph.edge_attr = tgutils.coalesce(new_graph.edge_index, new_graph.edge_attr, graph.num_nodes)
    return new_graph, anomalous_nodes

def attribute_perturb(graph: tgdata.Data, anomalous_nodes: torch.Tensor, k_nodes: int) -> tuple[tgdata.Data, torch.Tensor]:
    """
    Replaces the features of the anomalous nodes with the features of the farthest node in a set of k samples.

    Args:
        graph (tgdata.Data): The input graph.
        anomalous_nodes (torch.Tensor): The nodes to replace features for.
        k_nodes (int): The number of nodes to sample.

    Returns:
        tuple: A tuple containing the new graph and the anomalous nodes.
    """
    
    new_graph = graph.clone()
    # Sample k nodes for each anomalous node
    k_samples = torch.randint(0, graph.num_nodes, (anomalous_nodes.shape[0], k_nodes), device=graph.edge_index.device)
    # For each anamolous node, find the node in the k_samples that has the farthest l2 distance
    k_features = graph.x[k_samples] # Get the features of the k samples
    anomalous_features = graph.x[anomalous_nodes] # Get the features of the anomalous nodes
    # Get the l2 distance between the anomalous node and the k samples
    l2_distance = torch.linalg.norm(k_features - anomalous_features.unsqueeze(1), dim=2)
    # Get the index of the farthest node
    farthest_node = torch.argmax(l2_distance, dim=1)
    # Get the features of the farthest node
    farthest_features = k_features[torch.arange(k_samples.shape[0]), farthest_node]
    new_graph.x[anomalous_nodes] = farthest_features

    return new_graph, anomalous_nodes


def inject_anomalies(graph: tgdata.Data, 
                     percent_structural: float,
                     percent_attribute: float,
                     clique_size: int=5) -> tuple[tgdata.Data, torch.Tensor, torch.Tensor]:
    """
    Injects anomalies into the graph by creating cliques and replacing node features.

    Args:
        graph (tgdata.Data): The input graph.
        percent_structural (float): The percentage of structural anomalies to inject.
        percent_attribute (float): The percentage of attribute anomalies to inject.
        clique_size (int): The size of the cliques to create.

    Returns:
        tuple: A tuple containing the new graph, the structural anomalies, and the attribute anomalies.
    """
    
    # Calculate the number of structural and attribute anomalies
    num_structural = int(graph.num_nodes * percent_structural)
    num_attribute = int(graph.num_nodes * percent_attribute)
    total_anomalies = num_structural + num_attribute
    num_cliques = num_structural // clique_size
    if num_cliques * clique_size < num_structural:
        # If the number of structural anomalies is not divisible by the clique size, add
        # the remainder of the structural anomalies to the number of attribute anomalies
        num_attribute += num_structural - (num_cliques * clique_size)
        num_structural = num_cliques * clique_size
    
    print("Number of structural anomalies:", num_structural)
    print("Number of attribute anomalies:", num_attribute)
    print("Total anomalies:", total_anomalies)
    # Ensures all anomalous nodes are unique
    anomalous_nodes = torch.randperm(graph.num_nodes, device=graph.edge_index.device)[:total_anomalies]
    # Create the structural anomalies
    new_graph, structural_anomalies = structurally_perturb(graph, anomalous_nodes[:num_structural], clique_size)

    # Create the attribute anomalies
    new_graph, attribute_anomalies = attribute_perturb(new_graph, anomalous_nodes[num_structural:], 50)
    return new_graph, structural_anomalies, attribute_anomalies

def main():
    """
    Main function to inject anomalies into a graph, and save the results.
    """

    import os
    import argparse
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid

    parser = argparse.ArgumentParser(description="Anomaly Injection")
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    parser.add_argument("--percent_structural", type=float, default=0.025, help="Percentage of structural anomalies to inject")
    parser.add_argument("--percent_attribute", type=float, default=0.025, help="Percentage of attribute anomalies to inject")
    parser.add_argument("--clique_size", type=int, default=5, help="Size of the cliques to create")
    
    args = parser.parse_args()
    os.makedirs(f"perturbed_data/{args.dataset}", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    dataset = Planetoid(f"datasets", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Create the anomalies
    new_graph, struct_anomalies, att_anomalies = inject_anomalies(data, args.percent_structural, args.percent_attribute, args.clique_size)
    # Save the anomalies
    torch.save(new_graph, f"perturbed_data/{args.dataset}/perturbed_data.pt")
    torch.save(struct_anomalies, f"perturbed_data/{args.dataset}/struct_anomalies.pt")
    torch.save(att_anomalies, f"perturbed_data/{args.dataset}/att_anomalies.pt")
    with open(f"perturbed_data/{args.dataset}/anomalies.txt", "w") as f:
        f.write(f"Structural anomalies ({args.percent_structural * 100}%): {struct_anomalies.shape[0]}\n")
        f.write(f"\tClique size: {args.clique_size}\n")
        f.write(f"\tNumber of cliques: {struct_anomalies.shape[0] // args.clique_size}\n")
        f.write(f"Attribute anomalies ({args.percent_attribute * 100}%): {att_anomalies.shape[0]}\n")
        f.write(f"Total anomalies: {struct_anomalies.shape[0] + att_anomalies.shape[0]}\n")
    print(f"Anomalies saved to perturbed_data/{args.dataset}/anomalies.txt")

if __name__ == "__main__":
    main()