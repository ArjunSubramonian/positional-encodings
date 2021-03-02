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

def pre_process(d, args):
    # add summary node that connects to all the other nodes.
    # append row of -1's as raw features of summary node (modified AtomEncoder will specially handle all -1's)
    d.x = torch.cat([d.x, -torch.ones(1, d.x.size(1)).long()])
    node_size = d.x.size(0)
    # okay to add self-loop to summary node
    # don't need to coalesce
    # append columns to edge_index to connect summary node to all other nodes
    add_edges = torch.cat([(node_size - 1) * torch.ones(1, node_size).long(), \
                                torch.arange(node_size).long().reshape(1, -1)])
    d.edge_index = torch.cat([d.edge_index, add_edges], dim=1)
    # append rows of -1's as raw features of all new edges (modified BondEncoder will specially handle all -1's)
    d.edge_attr = torch.cat([d.edge_attr, -torch.ones(node_size, d.edge_attr.size(1)).long()])
    
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
    d.edge_index = dense_to_sparse(new_dense_orig_adj)[0]
    
    dense_extra_adj = new_dense_orig_adj - dense_orig_adj
    dense_orig_edge_attr[dense_extra_adj.bool()] = -2 * torch.ones(dense_orig_edge_attr.size(-1)).long()
    d.edge_attr = dense_orig_edge_attr[d.edge_index[0], d.edge_index[1]]
    
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


# +
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class ModifiedAtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, summary_node = True):
        super(ModifiedAtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        
        if summary_node:
            self.summary_node_embedding = torch.nn.Parameter(torch.empty(1, emb_dim))
            torch.nn.init.xavier_uniform_(self.summary_node_embedding.data)
        self.summary_node = summary_node

    def forward(self, x):
        mask = x.sum(dim=1) >= 0  # mask of all non-summary nodes
        
        x_embedding = 0
        for i in range(x[mask].shape[1]):
            x_embedding += self.atom_embedding_list[i](x[mask][:,i])
        
        mod_x_embedding = torch.empty(x.size(0), x_embedding.size(1), device=x.get_device())
        mod_x_embedding[mask] = x_embedding
        if self.summary_node:
            mod_x_embedding[~mask] = self.summary_node_embedding
        else:
            mod_x_embedding[~mask] = 0

        return mod_x_embedding

class ModifiedBondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim, dropout = 0.2, summary_node = True):
        super(ModifiedBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

        if summary_node:
            self.summary_link_embedding = torch.nn.Parameter(torch.empty(1, emb_dim))
            torch.nn.init.xavier_uniform_(self.summary_link_embedding.data)
        self.summary_node = summary_node
        
        self.drop = torch.nn.Dropout(dropout)
        
    def forward(self, edge_attr):
        mask = edge_attr.sum(dim=1) >= 0  # mask of all non-summary links
        mask_summary = edge_attr.sum(dim=1) == -edge_attr.size(1)
        mask_k_hops = edge_attr.sum(dim=1) == -2 * edge_attr.size(1) # k-hop neighbors, k > 1
        
        bond_embedding = 0
        for i in range(edge_attr[mask].shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[mask][:,i])
        
        mod_bond_embedding = torch.empty(edge_attr.size(0), bond_embedding.size(1), device=edge_attr.get_device())
        mod_bond_embedding[mask] = bond_embedding
        if self.summary_node:
            mod_bond_embedding[mask_summary] = self.summary_link_embedding
            mod_bond_embedding[mask_k_hops] = 0
        else:
            mod_bond_embedding[~mask] = 0
            
        return self.drop(mod_bond_embedding)
