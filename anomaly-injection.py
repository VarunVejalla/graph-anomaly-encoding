from random import randrange, sample
import torch
import torch_geometric.data as tgdata
import torch_geometric.utils as tgutils

def structurally_perturb(graph: tgdata.Data, m_nodes: int, n_cliques: int) -> tuple[tgdata.Data, torch.Tensor]:
    """
    Create a new graph by selecting m_nodes and creating a fully connected clique, repeat n_cliques times.
    
    Note: Anomalous nodes are returned as a tensor of shape (n_cliques, m_nodes), to get anomalous nodes, flatten
        the tensor and get the unqiue indexes.
    
    Args:
        graph (tgdata.Data): The original graph.
        m_nodes (int): The number of nodes in each clique.
        n_cliques (int): The number of cliques to create.
        
    Returns:
        tuple[tgdata.Data, torch.Tensor]: A tuple containing the new graph and the tensor of anomalous nodes.
    """ 
    
    new_graph = graph.clone() # May not be necessary be for now to maintain the original graph
    
    anomalous_nodes = torch.randint(0, graph.num_nodes, (n_cliques, m_nodes), device=graph.edge_index.device)
    
    # Alternative that ensures unique nodes *within* each clique, not across cliques
    # anomalous_nodes = torch.stack([torch.randperm(graph.num_nodes, device=graph.edge_index.device)[:m_nodes] for _ in range(n_cliques)])
    
    # Alternative that ensures all nodes are unique but requires m*n < num_nodes
    # anomalous_nodes = torch.randperm(graph.num_nodes, device=graph.edge_index.device)[:m_nodes * n_cliques].reshape(n_cliques, m_nodes)
    
    # Creates a set of tuples to represent the edges to prevent duplicates
    anomalous_edges = set()
    
    # Iterate through the cliques
    for clique in anomalous_nodes:
        # Iterate through the nodes in each clique
        for i, node in enumerate(clique):
            # Create edges between the nodes in the clique
            for j in range(i + 1, len(clique)):
                anomalous_edges.add((node.item(), clique[j].item())) # Pytorch represents undirected edges as (u, v) and (v, u)
                anomalous_edges.add((clique[j].item(), node.item())) # Add the reverse edge as well
                
    # Convert the set of edges back to a tensor
    anomalous_edges = torch.tensor(list(anomalous_edges), device=graph.edge_index.device).T
    new_graph.edge_index = torch.concat([new_graph.edge_index, anomalous_edges], dim=1)
    # Coalese sorts the edges and removes duplicates prevents having to convert to a list and set and back to a tensor
    new_graph.edge_index, new_graph.edge_attr = tgutils.coalesce(new_graph.edge_index, new_graph.edge_attr, graph.num_nodes, graph.num_nodes)
    
    return new_graph, anomalous_nodes

def attribute_perturb(graph: tgdata.Data, n_nodes: int) -> tuple[tgdata.Data, torch.Tensor]:
    """
    Samples n_nodes negative edges from the graph and replaces the features of the source nodes with the target nodes.
    
    Negative sampling is used to get attributes of the target nodes (at least not a neighboring node) and replace the
    source node's features with the target node's features. This will place the target node's features "out of place"
    because the target and source nodes are at least not connected, but could be even further apart.
    
    Args:
        graph (tgdata.Data): The original graph.
        n_nodes (int): The number of nodes to sample.
        
    Returns:
        tuple[tgdata.Data, torch.Tensor]: A tuple containing the new graph and the tensor of anomalous nodes.
        
    Raises:
        ValueError: If the graph does not have node features.
    """
    
    # NOTE: This is a tradeoff for time, instead of getting the node that has the farthest l2 distance, we just sample
    #      a random node that is not connected to the source node.
    
    sampled = tgutils.negative_sampling(graph.edge_index, num_nodes=graph.num_nodes, num_neg_samples=n_nodes)
    while sampled[0].unique().shape[0] != n_nodes:
        # Ensure that the src nodes are unique
        print("Resampling...")
        sampled = tgutils.negative_sampling(graph.edge_index, num_nodes=graph.num_nodes, num_neg_samples=n_nodes)
    
    # Anomalous nodes are the source nodes, other node's features are copeid to the anomalous nodes
    anomalous_nodes = sampled[0]
    normal_nodes = sampled[1]
    new_graph = graph.clone()
    
    # Check if the graph has node features
    if graph.x is None:
        raise ValueError("Graph does not have node features.")
    
    # Copy the node features of the normal nodes to the anomalous nodes
    new_graph.x[anomalous_nodes] = graph.x[normal_nodes]
    return new_graph, anomalous_nodes


def inject_anomalies(data, num_struct_anomalies, struct_anomaly_size, num_context_anomalies):
    # adds in anomalies
    # num_struct_anomalies * struct_anomaly_size == num_context_anomalies maybe
    
    # TODO: ensure no overlap between cliques? idk if this is needed or not
    for _ in range(num_struct_anomalies):
        clique = sample(range(data.num_nodes), struct_anomaly_size)
        
        # TODO
        for i in clique:
            for j in clique:
                if i == j:
                    continue
                
                # add edge from i to j
                data.put_edge_index()
    
    # TODO: should edges be undirected?
        
    
    # TODO: idk if there should be no overlap between structural anomalies and contextual anomalies either
    
    context_anomalies = set()
    for _ in range(data.num_nodes):
        # choose random index that wasn't chosen before
        ri = randrange(data.num_nodes)
        while ri in context_anomalies:
            ri = randrange(data.num_nodes)
        
        # TODO: perturb data.x[ri]
        
        context_anomalies.add(ri)
    
    return -1