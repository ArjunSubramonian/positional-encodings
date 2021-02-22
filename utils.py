import torch
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.approximation.connectivity import all_pairs_node_connectivity
from networkx.algorithms.clique import node_clique_number
from networkx.algorithms.centrality import betweenness_centrality, edge_betweenness_centrality
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

type_of_encoding = ["link", 'shortest_dist', "connectivity", 'edge_betweenness', "clique_number"]


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
    # print(k)
    max_value = 2
    connect_edge_index = torch.LongTensor(2, d.x.size(0) * d.x.size(0))
    connect_edge_attr = torch.FloatTensor(d.x.size(0) * d.x.size(0), 1)
    for i in range(d.x.size(0)):
        for j in range(d.x.size(0)):
            connect_edge_index[0][i * d.x.size(0) + j] = i
            connect_edge_index[1][i * d.x.size(0) + j] = j

            if not i == j:
                connect_edge_attr[i * d.x.size(0) + j] = k[i][j]
                if k[i][j] > max_value:
                    max_value = k[i][j]
    # print("enter")
    # print("connect_edge {}".format(connect_edge_index))
    # print("max value is {}".format(max_value))
    return Data(x=d.x, y=d.y, edge_index=d.edge_index, edge_attr=d.edge_attr, other_edge_index=connect_edge_index,
                other_edge_attr=connect_edge_attr)

# This is node level & edge level betweenness centrality


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
