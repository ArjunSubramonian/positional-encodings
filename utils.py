import torch
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.clique import node_clique_number
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality
from networkx.algorithms.link_prediction import resource_allocation_index, jaccard_coefficient,\
adamic_adar_index, preferential_attachment, cn_soundarajan_hopcroft, ra_index_soundarajan_hopcroft, within_inter_cluster, common_neighbor_centrality
from networkx.algorithms.communicability_alg import communicability
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

type_of_encoding = ["link", 'shortest_dist', "connectivity",
                    "jaccard", "com", "alloc",
                    "adamic", "preferential", "cn_soundarajan",
                    "ra_index", "centrality", "cluster"]

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
    connect_edge_index = torch.zeros(2, d.x.size(0) * d.x.size(0))
    connect_edge_attr = torch.zeros(d.x.size(0) * d.x.size(0), 1)

    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            connect_edge_index[0][i * d.x.size(0) + j] = i
            connect_edge_index[1][i * d.x.size(0) + j] = j

            if not i == j:
                connect_edge_attr[i * d.x.size(0) + j] = k[i][j]
    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, other_edge_index=connect_edge_index,
                other_edge_attr=connect_edge_attr)

# This is node level & edge level betweenness centrality

def tuple_fetch(size, tuple):
    current_max = 0
    index = torch.zeros(2, size * size)
    attr = torch.zeros(size * size, 1)

    for i, j, p in tuple:
        index[0][i * size + j] = i
        index[1][i * size + j] = j

        attr[i * size + j] = p
        if p > current_max:
            current_max = p

    return current_max, index, attr

### all other at once###

def compute_all_attributes(d):
    d_nx = to_networkx(d, to_undirected=True)
    com_dict = communicability(d_nx)
    com_max = 0
    com_index = torch.zeros(2, d.x.size(0) * d.x.size(0))
    com_attr = torch.zeros(d.x.size(0) * d.x.size(0), 1)

    alloc_t = resource_allocation_index(d_nx)
    jaccard_t = jaccard_coefficient(d_nx)
    adamic_t = adamic_adar_index(d_nx)
    preferential_t = preferential_attachment(d_nx)
    # cn_soundarajan_t = cn_soundarajan_hopcroft(d_nx)
    # ra_index_t = ra_index_soundarajan_hopcroft(d_nx)
    # centrality_t = common_neighbor_centrality(d_nx)
    # cluster_t = within_inter_cluster(d_nx)

    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            com_index[0][i * d.x.size(0) + j] = i
            com_index[1][i * d.x.size(0) + j] = j

            com_attr[i * d.x.size(0) + j] = com_dict[i][j]
            if com_dict[i][j] > com_max:
                com_max = com_dict[i][j]

    alloc_max, alloc_index, alloc_attr = tuple_fetch(d.x.size(0), alloc_t)
    jaccard_max, jaccard_index, jaccard_attr = tuple_fetch(d.x.size(0), jaccard_t)
    adamic_max, adamic_index, adamic_attr = tuple_fetch(d.x.size(0), adamic_t)
    preferential_max, preferential_index, preferential_attr = tuple_fetch(d.x.size(0), preferential_t)
    # cn_soundarajan_max, cn_soundarajan_index, cn_soundarajan_attr = tuple_fetch(d.x.size(0), cn_soundarajan_t)
    # ra_index_max, ra_index_index, ra_index_attr = tuple_fetch(d.x.size(0), ra_index_t)
    # centrality_max, centrality_index, centrality_attr = tuple_fetch(d.x.size(0), centrality_t)
    # cluster_max, cluster_index, cluster_attr = tuple_fetch(d.x.size(0), cluster_t)

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr,
                com_index=com_index, com_attr=com_attr, com_max=com_max,
                alloc_max=alloc_max, alloc_index=alloc_index, alloc_attr=alloc_attr,
                jaccard_max=jaccard_max, jaccard_index=jaccard_index, jaccard_attr=jaccard_attr,
                adamic_max=adamic_max, adamic_index=adamic_index, adamic_attr=adamic_attr,
                preferential_max=preferential_max, preferential_index=preferential_index, preferential_attr=preferential_attr,
                # cn_soundarajan_max=cn_soundarajan_max, cn_soundarajan_index=cn_soundarajan_index, cn_soundarajan_attr=cn_soundarajan_attr,
                # ra_index_max=ra_index_max, ra_index_index=ra_index_index, ra_index_attr=ra_index_attr,
                # centrality_max=centrality_max, centrality_index=centrality_index, centrality_attr=centrality_attr
                # cluster_max=cluster_max, cluster_index=cluster_index, cluster_attr=cluster_attr
                )


########################node level, don't use for now###################


def compute_edge_betweenness_centrality(d):
    d_nx = to_networkx(d, to_undirected=True)
    edge_dict = edge_betweenness_centrality(d_nx, k=5)

    bt_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.edge_index.size(1)):
        bt_edge_attr[i] = edge_dict[i]

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, other_edge_index=d.edge_index,
                other_edge_attr=bt_edge_attr)


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

<<<<<<< HEAD
    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, other_edge_index=cliq_edge_index,
                other_edge_attr=cliq_edge_attr)
=======
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
    #     TODO: add summary node that connects to all the other nodes.
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


# +
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class ModifiedAtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ModifiedAtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        
        self.summary_node_embedding = torch.nn.Parameter(torch.empty(1, emb_dim))
        torch.nn.init.xavier_uniform_(self.summary_node_embedding.data)

    def forward(self, x):
        mask = x.sum(dim=1) >= 0  # mask of all non-summary nodes
        
        x_embedding = 0
        for i in range(x[mask].shape[1]):
            x_embedding += self.atom_embedding_list[i](x[mask][:,i])
        
        mod_x_embedding = torch.empty(x.size(0), x_embedding.size(1), device=x.get_device())
        mod_x_embedding[mask] = x_embedding
        mod_x_embedding[~mask] = self.summary_node_embedding
    
        return mod_x_embedding

class ModifiedBondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim, dropout = 0.2):
        super(ModifiedBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

        self.summary_link_embedding = torch.nn.Parameter(torch.empty(1, emb_dim))
        torch.nn.init.xavier_uniform_(self.summary_link_embedding.data)
        
        self.drop = torch.nn.Dropout(dropout)
        
    def forward(self, edge_attr):
        mask = edge_attr.sum(dim=1) >= 0  # mask of all non-summary links
        
        bond_embedding = 0
        for i in range(edge_attr[mask].shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[mask][:,i])
        
        mod_bond_embedding = torch.empty(edge_attr.size(0), bond_embedding.size(1), device=edge_attr.get_device())
        mod_bond_embedding[mask] = bond_embedding
        mod_bond_embedding[~mask] = self.summary_link_embedding

        return self.drop(mod_bond_embedding)
>>>>>>> 5b89eccbde8928496c038613d926247ac3ccd88a
