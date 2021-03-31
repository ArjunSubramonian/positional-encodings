# +
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient, adamic_adar_index
from networkx.algorithms.communicability_alg import communicability
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.nn.functional as F

from sklearn.cluster import spectral_clustering
from scipy.sparse import csgraph


# -

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



def get_optimizer(model: nn.Module, learning_rate: float = 1e-4, adam_eps: float = 1e-6,
                  weight_decay: float = 0.0) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


# +
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

def one_hot(a, num_classes=None):
    if not num_classes:
        num_classes = a.max() + 1
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def robust_spec_cluster(adj, num_clusters):
    '''
    Helper function for hierarchical_shortest_distance
    
    Able to handle disconnected graphs
    first separate connected components, and then apply spectral clustering on the largest cc
    '''
    cc_label = csgraph.connected_components(adj)[1]
    unique_cc_label, unique_count = np.unique(cc_label, return_counts=True)
    num_cc = unique_cc_label.size
    mcc_size = unique_count.max()
    
    if num_cc == 1:
        if mcc_size <= num_clusters:
            return np.arange(mcc_size, dtype=np.int32)
        else:
            return spectral_clustering(adj, n_clusters=num_clusters)
    else:
        mcc_clusters = num_clusters - num_cc + 1
        if mcc_clusters <= 1: # num of cc is more than num clusters we desire
            return cc_label
        
        mcc_label = unique_count.argmax()
        mcc_mask = cc_label == mcc_label
        if mcc_size <= mcc_clusters: # num clusters is too large, each node forms cluster
            mcc_clusters = mcc_size
            mcc_pred = np.arange(mcc_size, dtype=np.int32)    
        else:
            mcc_pred = spectral_clustering(adj[mcc_mask][:,mcc_mask], n_clusters=mcc_clusters)
            
        for i, l in enumerate(unique_cc_label[unique_cc_label != mcc_label]):
            cc_label[cc_label == l] = i + mcc_clusters
        cc_label[mcc_mask] = mcc_pred

    return cc_label

def hierarchical_shortest_distance(node_size, dense_orig_adj, new_edge_index, hier_levels=3):
    adj = dense_orig_adj.float().numpy()
    
    node_assign = np.eye(node_size)
    sd = csgraph.shortest_path(adj).clip(0,100)
    edge_attr = [sd[new_edge_index[0], new_edge_index[1]].reshape(1, -1)]
    cluster_label = [np.arange(node_size, dtype=np.int32).reshape(1, -1)]

    num_clusters = node_size
    for i in range(hier_levels):
        if num_clusters // 3 > 1:
            num_clusters = num_clusters // 3
        elif num_clusters // 2 > 1:
            num_clusters = num_clusters // 2
        else:
            i -= 1
            break
        
        pred = robust_spec_cluster(adj, num_clusters)
        super_assign = one_hot(pred) # (N_i x N_{i+1}), assign super nodes to the next level super nodes
        adj = super_assign.transpose().dot(adj).dot(super_assign) # (N_{i+1} x N_i) * (N_i x N_i) *(N_i x N_{i+1})
        node_assign = node_assign.dot(super_assign) # (N_0 x N_i) * (N_i x N_{i+1}), assign original nodes to the next level super nodes
        super_sd = csgraph.shortest_path(adj.clip(0,1)).clip(0,100)
        sd = node_assign.dot(super_sd).dot(node_assign.transpose()) # (N_0 x N_i) * (N_i x N_i) * (N_i x N_0)

        cluster_label += [node_assign.argmax(axis=1).reshape(1, -1)]
        edge_attr += [sd[new_edge_index[0], new_edge_index[1]].reshape(1, -1)]
        
    for j in range(i + 1, hier_levels):
        cluster_label += [cluster_label[-1]]
        edge_attr += [edge_attr[-1]]
    
    hier_sd = torch.from_numpy(np.concatenate(edge_attr, axis=0)).t().long()
    hier_label = torch.from_numpy(np.concatenate(cluster_label, axis=0)).t().long()
    
    return hier_sd, hier_label


# -

def compute_all_attributes(d_nx, edge_index):
    attr_funcs = [communicability, resource_allocation_index, jaccard_coefficient, adamic_adar_index]
    attr_names = ['communicability', 'resource_allocation_index', 'jaccard_coefficient', 'adamic_adar_index']
    attrs = zip(attr_funcs, attr_names)
    
    edge_attrs = {}
    for func, name in attrs:
        edge_attr = []
        p = func(d_nx)
        for s, t in edge_index.t().tolist():
            if s in p and t in p[s]:
                edge_attr += [p[s][t]]
            else:
                edge_attr += [0]
        edge_attrs[name] = (torch.LongTensor(edge_attr))

    return edge_attrs


# +
def pre_process(d, args):
    node_size = d.x.size(0)
    
    #     Construct networkX type of original graph for different metrics
    d_nx = to_networkx(d, to_undirected=True)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, max_num_nodes=node_size).squeeze(dim=0).long()
    dense_orig_edge_attr = to_dense_adj(d.edge_index, edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
    pow_dense_orig_adj = dense_orig_adj.clone()
    new_dense_orig_adj = dense_orig_adj.clone()
    for k in range(2, args.k_hop_neighbors + 1):
        pow_dense_orig_adj = torch.mm(pow_dense_orig_adj, dense_orig_adj)
        new_dense_orig_adj |= F.hardtanh(pow_dense_orig_adj)
    new_edge_index = dense_to_sparse(new_dense_orig_adj)[0]
    
    dense_extra_adj = new_dense_orig_adj - dense_orig_adj
    dense_orig_edge_attr[dense_extra_adj.bool()] = -2 * torch.ones(dense_orig_edge_attr.size(-1)).long()
    new_edge_attr = dense_orig_edge_attr[new_edge_index[0], new_edge_index[1]]
    
    #     Calculate structural feature by the ORIGNAL graph, add them to new edge set.
    sd_edge_attr = shortest_distances(d_nx, new_edge_index).view(-1, 1)
    cn_edge_attr = node_connectivity(d_nx, new_edge_index).view(-1, 1)
    hsd_edge_attr, hier_label = hierarchical_shortest_distance(node_size, dense_orig_adj, new_edge_index, args.hier_levels)
    other_edge_attrs = compute_all_attributes(d_nx, new_edge_index)
    
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, orig_edge_index=d.edge_index, edge_attr=new_edge_attr, \
         orig_edge_attr=d.edge_attr, sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr, hsd_edge_attr=hsd_edge_attr, hier_label=hier_label, \
                comm_edge_attr=other_edge_attrs['communicability'], \
                alloc_edge_attr=other_edge_attrs['resource_allocation_index'], \
                jaccard_edge_attr=other_edge_attrs['jaccard_coefficient'], \
                adamic_edge_attr=other_edge_attrs['adamic_adar_index'])

def pre_process_with_summary(d, args):
    node_size = d.x.size(0)
    
    #     Construct networkX type of original graph for different metrics
    d_nx = to_networkx(d, to_undirected=True)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, max_num_nodes=node_size).squeeze(dim=0).long()
    dense_orig_edge_attr = to_dense_adj(d.edge_index, edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
    pow_dense_orig_adj = dense_orig_adj.clone()
    new_dense_orig_adj = dense_orig_adj.clone()
    for k in range(2, args.k_hop_neighbors + 1):
        pow_dense_orig_adj = torch.mm(pow_dense_orig_adj, dense_orig_adj)
        new_dense_orig_adj |= F.hardtanh(pow_dense_orig_adj)
    new_edge_index = dense_to_sparse(new_dense_orig_adj)[0]
    
    dense_extra_adj = new_dense_orig_adj - dense_orig_adj
    dense_orig_edge_attr[dense_extra_adj.bool()] = -2 * torch.ones(dense_orig_edge_attr.size(-1)).long()
    new_edge_attr = dense_orig_edge_attr[new_edge_index[0], new_edge_index[1]]
    
    #     Calculate structural feature by the ORIGNAL graph, add them to new edge set.
    sd_edge_attr = shortest_distances(d_nx, new_edge_index)
    cn_edge_attr = node_connectivity(d_nx, new_edge_index)
    hsd_edge_attr, hier_label = hierarchical_shortest_distance(node_size, dense_orig_adj, new_edge_index, args.hier_levels)
    other_edge_attrs = compute_all_attributes(d_nx, new_edge_index)
    
    # add summary node that connects to all the other nodes.
    # append row of -1's as raw features of summary node (modified AtomEncoder will specially handle all -1's)
    d.x = torch.cat([d.x, -torch.ones(1, d.x.size(1)).long()])
    # okay to add self-loop to summary node
    # don't need to coalesce
    # append columns to edge_index to connect summary node to all other nodes
    summary_src = node_size * torch.ones(1, node_size).long()
    summary_tgt = torch.arange(node_size).reshape(1, -1).long()
    summary_edges = torch.cat([summary_src, summary_tgt])
    summary_edges = torch.cat([summary_edges, summary_edges[torch.LongTensor([1,0])]], dim=1) 
    
    new_edge_index = torch.cat([new_edge_index, summary_edges], dim=1)
    # append rows of -1's as raw features of all new edges (modified BondEncoder will specially handle all -1's)
    new_edge_attr = torch.cat([new_edge_attr, -torch.ones(node_size*2, new_edge_attr.size(1)).long()])
    sd_edge_attr = torch.cat([sd_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    cn_edge_attr = torch.cat([cn_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    for attr_name in other_edge_attrs:
        other_edge_attrs[attr_name] = torch.cat([other_edge_attrs[attr_name], -torch.ones(node_size*2).long()]).view(-1, 1)
    
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, orig_edge_index=d.edge_index, edge_attr=new_edge_attr, \
         orig_edge_attr=d.edge_attr, sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr, hsd_edge_attr=hsd_edge_attr, hier_label=hier_label, \
                comm_edge_attr=other_edge_attrs['communicability'], \
                alloc_edge_attr=other_edge_attrs['resource_allocation_index'], \
                jaccard_edge_attr=other_edge_attrs['jaccard_coefficient'], \
                adamic_edge_attr=other_edge_attrs['adamic_adar_index'])

# +


# def pre_process(d, args):
#     node_size = d.x.size(0)
    
#     #     Construct networkX type of original graph for different metrics
#     d_nx = to_networkx(d, to_undirected=True)
    
#     #     Augment the graph to be K-hop graph
#     dense_orig_adj = to_dense_adj(d.edge_index, max_num_nodes=node_size).squeeze(dim=0).long()
#     dense_orig_edge_attr = to_dense_adj(d.edge_index, edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
#     pow_dense_orig_adj = dense_orig_adj.clone()
#     new_dense_orig_adj = dense_orig_adj.clone()
#     for k in range(2, args.k_hop_neighbors + 1):
#         pow_dense_orig_adj = torch.mm(pow_dense_orig_adj, dense_orig_adj)
#         new_dense_orig_adj |= F.hardtanh(pow_dense_orig_adj)
#     d.edge_index = dense_to_sparse(new_dense_orig_adj)[0]
    
#     dense_extra_adj = new_dense_orig_adj - dense_orig_adj
#     dense_orig_edge_attr[dense_extra_adj.bool()] = -2 * torch.ones(dense_orig_edge_attr.size(-1)).long()
#     d.edge_attr = dense_orig_edge_attr[d.edge_index[0], d.edge_index[1]]
    
#     #     Calculate structural feature by the ORIGNAL graph, add them to new edge set.
#     sd_edge_attr = shortest_distances(d_nx, d.edge_index)
#     cn_edge_attr = node_connectivity(d_nx, d.edge_index)
    
#     # add summary node that connects to all the other nodes.
#     # append row of -1's as raw features of summary node (modified AtomEncoder will specially handle all -1's)
#     d.x = torch.cat([d.x, -torch.ones(1, d.x.size(1)).long()])
#     # okay to add self-loop to summary node
#     # don't need to coalesce
#     # append columns to edge_index to connect summary node to all other nodes
#     summary_src = node_size * torch.ones(1, node_size).long()
#     summary_tgt = torch.arange(node_size).long().reshape(1, -1)
#     summary_edges = torch.cat([summary_src, summary_tgt])
#     summary_edges = torch.cat([summary_edges, summary_edges[torch.LongTensor([1,0])]], dim=1) 
#     add_edges = summary_edges
# #     print('add_edges', add_edges)
    
#     d.edge_index = torch.cat([d.edge_index, add_edges], dim=1)
#     # append rows of -1's as raw features of all new edges (modified BondEncoder will specially handle all -1's)
#     d.edge_attr = torch.cat([d.edge_attr, -torch.ones(node_size*2, d.edge_attr.size(1)).long()])
#     sd_edge_attr = torch.cat([sd_edge_attr, -torch.ones(node_size*2).long()])
#     cn_edge_attr = torch.cat([cn_edge_attr, -torch.ones(node_size*2).long()])
    
#     return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, \
#          sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr)
