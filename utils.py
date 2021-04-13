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
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
import torch.nn.functional as F

from sklearn.cluster import spectral_clustering
from scipy.sparse import csgraph

from scipy.linalg import eigh
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
import math


# -

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


# +
def compute_communicability(d_nx, edge_index):
    edge_attr = []
    p = communicability(d_nx)
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [p[s][t]]
        else:
            edge_attr += [0]
    return torch.FloatTensor(edge_attr)

def compute_resource_allocation_index(d_nx, edge_index):
    edge_attr = []
    r = resource_allocation_index(d_nx)
    p = {}
    for u, v, q in r:
        if u not in p:
            p[u] = {}
        p[u][v] = q
    
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [p[s][t]]
        else:
            edge_attr += [0]
    return torch.FloatTensor(edge_attr)

def compute_jaccard_coefficient(d_nx, edge_index):
    edge_attr = []
    r = jaccard_coefficient(d_nx)
    p = {}
    for u, v, q in r:
        if u not in p:
            p[u] = {}
        p[u][v] = q
    
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [p[s][t]]
        else:
            edge_attr += [0]
    return torch.FloatTensor(edge_attr)

def compute_adamic_adar_index(d_nx, edge_index):
    edge_attr = []
    r = adamic_adar_index(d_nx)
    p = {}
    for u, v, q in r:
        if u not in p:
            p[u] = {}
        p[u][v] = q
    
    for s, t in edge_index.t().tolist():
        if s in p and t in p[s]:
            edge_attr += [p[s][t]]
        else:
            edge_attr += [0]
    return torch.FloatTensor(edge_attr)


# -

def positionalencoding1d(edge_attr, d_model):
    with torch.no_grad():
        pe = torch.zeros(edge_attr.size(0), d_model, device=edge_attr.device)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=edge_attr.device) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(edge_attr * div_term)
        pe[:, 1::2] = torch.cos(edge_attr * div_term)

    return pe


def laplacian_pos_encodings(d_nx, k):
    sp_lap = normalized_laplacian_matrix(d_nx)
    N = d_nx.number_of_nodes()

    _, eigvecs = eigh(sp_lap.toarray()) # all eigenvectors
    if k + 1 > N:
        return torch.cat([torch.Tensor(eigvecs[:, 1:]), torch.zeros(N, k - N + 1)], dim=1) # remove nontrivial eigenvector and pad
    else:
        return torch.Tensor(eigvecs[:, 1:k+1]) # remove nontrivial eigenvector


# +
def pre_process(d, args):
    node_size = d.x.size(0)
    
    #     Construct networkX type of original graph for different metrics
    d_nx = to_networkx(d, to_undirected=True)
    d_nx_without_self_loops = to_networkx(d, to_undirected=True, remove_self_loops=True)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), max_num_nodes=node_size).squeeze(dim=0).long()
    dense_orig_edge_attr = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
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
    comm_edge_attr = compute_communicability(d_nx, new_edge_index).view(-1, 1)
    alloc_edge_attr = compute_resource_allocation_index(d_nx, new_edge_index).view(-1, 1)
    jaccard_edge_attr = compute_jaccard_coefficient(d_nx, new_edge_index).view(-1, 1)
    adamic_edge_attr = compute_adamic_adar_index(d_nx, new_edge_index).view(-1, 1)
    
    laplacian_pos = laplacian_pos_encodings(d_nx_without_self_loops, args.lap_k)
    assert laplacian_pos.size(0) == d_nx.number_of_nodes() and laplacian_pos.size(1) == args.lap_k
    
    return Data(x=d.x, lap_x=laplacian_pos, y=d.y, edge_index=new_edge_index, orig_edge_index=d.edge_index, edge_attr=new_edge_attr, \
         orig_edge_attr=d.edge_attr, sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr, hsd_edge_attr=hsd_edge_attr, hier_label=hier_label, \
                comm_edge_attr=comm_edge_attr, \
                alloc_edge_attr=alloc_edge_attr, \
                jaccard_edge_attr=jaccard_edge_attr, \
                adamic_edge_attr=adamic_edge_attr)

def pre_process_with_summary(d, args):
    node_size = d.x.size(0)
    
    #     Construct networkX type of original graph for different metrics
    d_nx = to_networkx(d, to_undirected=True)
    d_nx_without_self_loops = to_networkx(d, to_undirected=True, remove_self_loops=True)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), max_num_nodes=node_size).squeeze(dim=0).long()
    dense_orig_edge_attr = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
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
    comm_edge_attr = compute_communicability(d_nx, new_edge_index)
    alloc_edge_attr = compute_resource_allocation_index(d_nx, new_edge_index)
    jaccard_edge_attr = compute_jaccard_coefficient(d_nx, new_edge_index)
    adamic_edge_attr = compute_adamic_adar_index(d_nx, new_edge_index)
    
    laplacian_pos = laplacian_pos_encodings(d_nx_without_self_loops, args.lap_k)
    assert laplacian_pos.size(0) == d_nx.number_of_nodes() and laplacian_pos.size(1) == args.lap_k
    
    # add summary node that connects to all the other nodes.
    # append row of -1's as raw features of summary node (modified AtomEncoder will specially handle all -1's)
    d.x = torch.cat([d.x, -torch.ones(1, d.x.size(1)).long()])
    laplacian_pos = torch.cat([laplacian_pos, torch.zeros(1, laplacian_pos.size(1)).long()])
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
    hsd_edge_attr = torch.cat([hsd_edge_attr, -torch.ones(node_size*2, hsd_edge_attr.size(1)).long()])
    comm_edge_attr = torch.cat([comm_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    alloc_edge_attr = torch.cat([alloc_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    jaccard_edge_attr = torch.cat([jaccard_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    adamic_edge_attr = torch.cat([adamic_edge_attr, -torch.ones(node_size*2).long()]).view(-1, 1)
    
    return Data(x=d.x, lap_x=laplacian_pos, y=d.y, edge_index=new_edge_index, orig_edge_index=d.edge_index, edge_attr=new_edge_attr, \
         orig_edge_attr=d.edge_attr, sd_edge_attr=sd_edge_attr, cn_edge_attr=cn_edge_attr, hsd_edge_attr=hsd_edge_attr, hier_label=hier_label, \
                comm_edge_attr=comm_edge_attr, \
                alloc_edge_attr=alloc_edge_attr, \
                jaccard_edge_attr=jaccard_edge_attr, \
                adamic_edge_attr=adamic_edge_attr)


# -

def basic_pre_process_with_summary(d, args):
    node_size = d.x.size(0)
    
    #     Augment the graph to be K-hop graph
    dense_orig_adj = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), max_num_nodes=node_size).squeeze(dim=0).long()
    dense_orig_edge_attr = to_dense_adj(d.edge_index, batch=d.edge_index.new_zeros(node_size), edge_attr=d.edge_attr, max_num_nodes=node_size).squeeze(dim=0).long()
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
    inv_deg = 1 / degree(d.edge_index[0], node_size)
    inv_deg[inv_deg == float("inf")] = 0 # eliminate inf's
    pow_rw_edge_attr = torch.eye(node_size)
    rw_edge_attr = torch.mm(dense_orig_adj.float(), torch.diag(inv_deg))
    rw_edge_attrs = {}
    for k in range(1, args.k_hop_neighbors + 1):
        pow_rw_edge_attr = torch.mm(pow_rw_edge_attr, rw_edge_attr)
        dense_extra_adj = new_dense_orig_adj - torch.where(pow_rw_edge_attr == 0, pow_rw_edge_attr, torch.tensor(1.0)).long()
        pow_rw_edge_attr_clone = pow_rw_edge_attr.clone()
        pow_rw_edge_attr_clone[dense_extra_adj.bool()] = -1.0
        rw_edge_attrs['rw_edge_attr_' + str(k)] = dense_to_sparse(pow_rw_edge_attr_clone)[1]
    
    assert new_edge_attr.size(0) == rw_edge_attrs['rw_edge_attr_1'].size(0), inv_deg
      
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
    for k in rw_edge_attrs:
        rw_edge_attrs[k] = torch.cat([rw_edge_attrs[k], -torch.ones(node_size*2).long()]).view(-1, 1)
    
    return Data(x=d.x, y=d.y, edge_index=new_edge_index, orig_edge_index=d.edge_index, edge_attr=new_edge_attr, \
         orig_edge_attr=d.edge_attr, **rw_edge_attrs)


# +
import networkx as nx

'''
Graph drawing utils
'''
element_color_list = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'silver']
atom_color_map = {5:'silver', 6:'red', 7: 'blue', 15: 'yellow', 16: 'green'}

def get_node_color_list(g, element_color_list):
    node_color_list = []
    for n in g.nodes:
        idx = g.nodes[n]['element_id']
        idx = min(len(element_color_list) - 1, idx)
        node_color_list += [element_color_list[idx]]
    return node_color_list

def get_atom_color_list(g, atom_color_map):
    node_color_list = []
    for n in g.nodes:
        idx = g.nodes[n]['x'][0]
        node_color_list += [atom_color_map.get(idx, "black")]

    return node_color_list

def draw_with_color(g, ax=None, labels='none'):
    node_color_list = get_node_color_list(g, element_color_list)
    if labels == 'none':
        nx.draw(g, ax=ax, node_color=node_color_list)                    
    elif labels == 'node_id':
        nx.draw(g, ax=ax, node_color=node_color_list, with_labels=True)    
    else:
        nx.draw(g, ax=ax, node_color=node_color_list, labels=nx.get_node_attributes(g, labels))   
        
def molecule_draw_with_color(g, ax=None, labels='none'):
    node_color_list = get_atom_color_list(g, atom_color_map)
    if labels == 'none':
        nx.draw(g, ax=ax, node_color=node_color_list)                    
    elif labels == 'node_id':
        nx.draw(g, ax=ax, node_color=node_color_list, with_labels=True)    
    else:
        nx.draw(g, ax=ax, node_color=node_color_list, labels=nx.get_node_attributes(g, labels))   

# +
# from ogb.graphproppred import PygGraphPropPredDataset
# import argparse

# dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
# parser = argparse.ArgumentParser(description='PyTorch implementation of relative positional encodings and relation-aware self-attention for graph Transformers')
# args = parser.parse_args("")
# args.k_hop_neighbors = 3

# for d in dataset[:1]:
#     d_new = basic_pre_process_with_summary(d, args)
#     print(d_new)
#     print(d_new.rw_edge_attr_1)
#     print(d_new.rw_edge_attr_2)
#     print(d_new.rw_edge_attr_3)
# -

# ## Old Code

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

# +
# def compute_clique_number(d):
#     d_nx = to_networkx(d, to_undirected=True)
#     k = node_clique_number(d_nx)

#     cliq_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
#     cliq_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
#     for i in range(d.x.size(0)):
#         for j in range(d.x.size(0)):
#             cliq_edge_index[0][i * d.x.size(0) + j] = i
#             cliq_edge_index[1][i * d.x.size(0) + j] = j

#             cliq_edge_attr[i * d.x.size(0) + j] = k[i][j]

#     return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, cliq_edge_index=cliq_edge_index,
#                 cliq_edge_attr=cliq_edge_attr)

# +
# def compute_edge_betweenness_centrality(d):
#     d_nx = to_networkx(d, to_undirected=True)
#     edge_dict = edge_betweenness_centrality(d_nx, k=5)

#     bt_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
#     for i in range(d.edge_index.size(1)):
#         bt_edge_attr[i] = edge_dict[i]

#     return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, bt_edge_index=d.edge_index,
#                 bt_edge_attr=bt_edge_attr)

# +
# def compute_all_node_connectivity(d):
#     d_nx = to_networkx(d, to_undirected=True)
#     k = all_pairs_node_connectivity(d_nx)

#     connect_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
#     connect_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
#     for i in range(d.x.size(0)):
#         for j in range(d.x.size(0)):
#             connect_edge_index[0][i * d.x.size(0) + j] = i
#             connect_edge_index[1][i * d.x.size(0) + j] = j

#             connect_edge_attr[i * d.x.size(0) + j] = k[i][j]

#     return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, connect_edge_index=connect_edge_index,
#                 connect_edge_attr=connect_edge_attr)

# +
# def compute_mutual_shortest_distances(d):
#     d_nx = to_networkx(d, to_undirected=True)
#     p = shortest_path(d_nx)
    
#     sd_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
#     sd_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
#     for i in range(d.x.size(0)):
#         for j in range(d.x.size(0)):
#             sd_edge_index[0][i * d.x.size(0) + j] = i
#             sd_edge_index[1][i * d.x.size(0) + j] = j
            
#             if j in p[i]:
#                 sd_edge_attr[i * d.x.size(0) + j] = len(p[i][j]) - 1
#             else:
#                 sd_edge_attr[i * d.x.size(0) + j] = float("inf")
        
#     return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, sd_edge_index=sd_edge_index, sd_edge_attr=sd_edge_attr)

# +
# def laplacian_pos_encodings(d_nx, k):
#     sp_lap = normalized_laplacian_matrix(d_nx)
#     N = d_nx.number_of_nodes()
#     if N <= k + 2:
#         _, eigvecs = eigs(sp_lap, k=N-2, which='SM') # all smallest eigenvectors
#         return torch.cat([torch.Tensor(eigvecs[:, 1:]), torch.zeros(N, k - N + 3)], dim=1) # remove nontrivial eigenvector and pad
#     else:
#         _, eigvecs = eigs(sp_lap, k=k+1, which='SM') # k + 1 smallest eigenvectors
#         return torch.Tensor(eigvecs[:, 1:]) # remove nontrivial eigenvector
