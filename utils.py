import torch
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.clique import node_clique_number
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality
from networkx.algorithms.link_prediction.resource_allocation_index import resource_allocation_index
from networkx.algorithms.centrality.communicability_alg import communicability
from networkx.algorithms.link_prediction.jaccard_coefficient import jaccard_coefficient
from networkx.algorithms.link_prediction.adamic_adar_index import adamic_adar_index
from networkx.algorithms.link_prediction.preferential_attachment import preferential_attachment
from networkx.algorithms.link_prediction.cn_soundarajan_hopcroft import cn_soundarajan_hopcroft
from networkx.algorithms.link_prediction.ra_index_soundarajan_hopcroft import ra_index_soundarajan_hopcroft
from networkx.algorithms.link_prediction.within_inter_cluster import within_inter_cluster
from networkx.algorithms.link_prediction.common_neighbor_centrality import common_neighbor_centrality
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
    cn_soundarajan_t = cn_soundarajan_hopcroft(d_nx)
    ra_index_t = ra_index_soundarajan_hopcroft(d_nx)
    centrality_t = common_neighbor_centrality(d_nx)
    cluster_t = within_inter_cluster(d_nx)

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
    cn_soundarajan_max, cn_soundarajan_index, cn_soundarajan_attr = tuple_fetch(d.x.size(0), cn_soundarajan_t)
    ra_index_max, ra_index_index, ra_index_attr  = tuple_fetch(d.x.size(0), ra_index_t)
    centrality_max, centrality_index, centrality_attr = tuple_fetch(d.x.size(0), centrality_t)
    cluster_max, cluster_index, cluster_attr = tuple_fetch(d.x.size(0), cluster_t)

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr,
                com_index=com_index, com_attr=com_attr, com_max=com_max,
                alloc_max=alloc_max, alloc_index=alloc_index, alloc_attr=alloc_attr,
                jaccard_max=jaccard_max, jaccard_index=jaccard_index, jaccard_attr=jaccard_attr,
                adamic_max=adamic_max, adamic_index=adamic_index, adamic_attr=adamic_attr,
                preferential_max=preferential_max, preferential_index=preferential_index, preferential_attr=preferential_attr,
                cn_soundarajan_max=cn_soundarajan_max, cn_soundarajan_index=cn_soundarajan_index, cn_soundarajan_attr=cn_soundarajan_attr,
                ra_index_max=ra_index_max, ra_index_index=ra_index_index, ra_index_attr=ra_index_attr,
                centrality_max=centrality_max, centrality_index=centrality_index, centrality_attr=centrality_attr,
                cluster_max=cluster_max, cluster_index=cluster_index, cluster_attr=cluster_attr
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

    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, other_edge_index=cliq_edge_index,
                other_edge_attr=cliq_edge_attr)
