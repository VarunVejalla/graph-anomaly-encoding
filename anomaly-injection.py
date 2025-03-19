from random import randrange, sample
import torch
import torch_geometric.data as tgdata

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
    
    # Creates a set of tuples to represent the edges to prevent duplicates
    graph_edges = set(map(tuple, new_graph.edge_index.T.tolist()))
    
    # Iterate through the cliques
    for clique in anomalous_nodes:
        # Iterate through the nodes in each clique
        for i, node in enumerate(clique):
            # Create edges between the nodes in the clique
            for j in range(i + 1, len(clique)):
                graph_edges.add((node.item(), clique[j].item())) # Pytorch represents undirected edges as (u, v) and (v, u)
                graph_edges.add((clique[j].item(), node.item())) # Add the reverse edge as well
                
    # Convert the set of edges back to a tensor
    new_graph.edge_index = torch.tensor(list(graph_edges), dtype=torch.long).T
    
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