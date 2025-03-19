from random import randrange, sample

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