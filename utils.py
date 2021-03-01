import torch
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.clique import node_clique_number
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

def compute_mutual_shortest_distances(d):
    d_nx = to_networkx(d, to_undirected=True)
    p = shortest_path(d_nx)
    
    sd_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
    sd_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            sd_edge_index[0][i * d.x.size(0) + j] = i
            sd_edge_index[1][i * d.x.size(0) + j] = j
            
            if j in p[i]:
                sd_edge_attr[i * d.x.size(0) + j] = len(p[i][j]) - 1
            else:
                sd_edge_attr[i * d.x.size(0) + j] = float("inf")
        
    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, sd_edge_index=sd_edge_index, sd_edge_attr=sd_edge_attr)


def compute_all_node_connectivity(d):
    d_nx = to_networkx(d, to_undirected=True)
    k = all_pairs_node_connectivity(d_nx)

    connect_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
    connect_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            connect_edge_index[0][i * d.x.size(0) + j] = i
            connect_edge_index[1][i * d.x.size(0) + j] = j

            connect_edge_attr[i * d.x.size(0) + j] = k[i][j]

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, connect_edge_index=connect_edge_index,
                connect_edge_attr=connect_edge_attr)

# This is node level & edge level betweenness centrality

def compute_edge_betweenness_centrality(d):
    d_nx = to_networkx(d, to_undirected=True)
    edge_dict = edge_betweenness_centrality(d_nx, k=5)

    bt_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.edge_index.size(1)):
        bt_edge_attr[i] = edge_dict[i]

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, bt_edge_index=d.edge_index,
                bt_edge_attr=bt_edge_attr)


def compute_clique_number(d):
    d_nx = to_networkx(d, to_undirected=True)
    k = node_clique_number(d_nx)

    cliq_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
    cliq_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            cliq_edge_index[0][i * d.x.size(0) + j] = i
            cliq_edge_index[1][i * d.x.size(0) + j] = j

            cliq_edge_attr[i * d.x.size(0) + j] = k[i][j]

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, cliq_edge_index=cliq_edge_index,
                cliq_edge_attr=cliq_edge_attr)


# +
import torch
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.clique import node_clique_number
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.nn.functional as F

K_HOP_NEIGHBORS = 6

def pre_process(d):
    node_size = d.x.size(0)
    #     TODO: add summary node that connects to all the other nodes.
    
    #     Construct networkX type of original graph for different metrics
    d_nx = to_networkx(d, to_undirected=True)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, max_num_nodes=node_size).squeeze(dim=0).long()
    pow_dense_orig_adj = dense_orig_adj.clone()
    new_dense_orig_adj = dense_orig_adj.clone()
    for k in range(2, K_HOP_NEIGHBORS + 1):
        pow_dense_orig_adj = torch.mm(pow_dense_orig_adj, dense_orig_adj)
        new_dense_orig_adj |= F.hardtanh(pow_dense_orig_adj)
    d.edge_index = dense_to_sparse(new_dense_orig_adj)[0]
    
    #     Calculate structural feature by the ORIGNAL graph, add them to new edge set.
    sd_edge_attr = shortest_distances(d_nx, d.edge_index)
    cn_edge_attr = node_connectivity(d_nx, d.edge_index)
    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, \
         sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr)
    
def shortest_distances(d_nx, edge_index):
    edge_attr = []
    p = shortest_path(d_nx)
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [len(p[s][t]) - 1]
        else:
            edge_attr += [0]
        
    return torch.LongTensor(edge_attr)

def node_connectivity(d_nx, edge_index):
    edge_attr = []
    p = all_pairs_node_connectivity(d_nx)
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [p[s][t]]
        else:
            edge_attr += [0]
    return torch.LongTensor(edge_attr)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
