import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

class GraphTransformer(nn.Module):

    def __init__(self, layers, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GraphTransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout))
    
    def forward(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None):
        for idx, layer in enumerate(self.layers):
            x, _ = layer(x, relation, kv, self_padding_mask, self_attn_mask)
        return x

    def get_attn_weights(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None):
        attns = []
        for idx, layer in enumerate(self.layers):
            x, attn = layer(x, relation, kv, self_padding_mask, self_attn_mask, need_weights=True)
            attns.append(attn)
        attn = torch.stack(attns)
        return attn

class GraphTransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        # self.attn_batch_norm = nn.BatchNorm1d(embed_dim) # can consider BatchNorm in the future?
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        # self.ff_batch_norm = nn.BatchNorm1d(embed_dim) # can consider BatchNorm in the future?
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None,
                need_weights = False):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask, need_weights=need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn

class RelationMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(RelationMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
        ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.unsqueeze(1) + ra
        k = k.unsqueeze(0) + rb
        q *= self.scaling
        # q: tgt_len x src_len x bsz*heads x dim
        # k: tgt_len x src_len x bsz*heads x dim
        # v: src_len x bsz*heads x dim

        attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k])
        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if attn_mask is not None:
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(-1),
                float('-inf')
            )
            # will produce NaNs post-softmax because padding nodes are not connected to any other nodes
            # will get rid of NaNs later by setting them to 0
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)


        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)  # get rid of NaNs!

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights: tgt_len x src_len x bsz*heads
        # v: src_len x bsz*heads x dim
        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads 
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

        return output

class GraphTransformerModel(nn.Module):

    def __init__(self, args):
        super(GraphTransformerModel, self).__init__()
        self.model_type = 'GraphTransformerModel'
        self.encoder = AtomEncoder(emb_dim=args.embed_dim)
        self.edge_encoder = BondEncoder(emb_dim=args.embed_dim)
        self.transformer = GraphTransformer(args.graph_layers, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.weights_dropout)
        
        #Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.graph_pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.graph_pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.graph_pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.task_pred = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim),
            nn.BatchNorm1d(args.embed_dim),
            nn.ReLU(),
            nn.Linear(args.embed_dim, args.n_classes)
        )
        
        self.relation_type = args.relation_type
        self.max_vocab = args.max_vocab
        self.relation_encoder = nn.Embedding(args.max_vocab, args.embed_dim)
        
        self.k = args.k_hop_neighbors
        
    def forward(self, src, src_mask=None):
        x, mask = to_dense_batch(self.encoder(src.x), batch=src.batch, fill_value=0)
        x = x.transpose(0, 1)
        
        if self.k is not None:
            dense_orig_adj = to_dense_adj(src.edge_index, batch=src.batch, max_num_nodes=x.size(0)).squeeze(dim=-1)
            self_attn_mask = dense_orig_adj.bool()
            for k in range(2, self.k + 1):
                dense_orig_adj = torch.matmul(dense_orig_adj, dense_orig_adj)
                self_attn_mask |= dense_orig_adj.bool()
            self_attn_mask = ~self_attn_mask.permute(2, 1, 0)
        else:
            self_attn_mask = None
        
        if self.relation_type == 'link':
            relation = self.relation_encoder(to_dense_adj(src.edge_index, batch=src.batch, max_num_nodes=x.size(0)).long())
        elif self.relation_type == 'shortest_dist':
            mod_sd_edge_attr = torch.clamp(src.sd_edge_attr.reshape(-1), 0, self.max_vocab - 1).long()
            relation = self.relation_encoder(to_dense_adj(src.sd_edge_index, batch=src.batch, edge_attr=mod_sd_edge_attr, max_num_nodes=x.size(0)).long())
        else:
            raise ValueError("Invalid relation type.")
        
        # integrate given edge features
        mod_edge_attr = self.edge_encoder(src.edge_attr)
        relation += to_dense_adj(src.edge_index, batch=src.batch, edge_attr=mod_edge_attr, max_num_nodes=x.size(0))
        relation = relation.permute(2, 1, 0, 3)
        
        output = self.transformer(x, relation, self_padding_mask=~mask.transpose(0, 1), self_attn_mask=self_attn_mask)
        
        graph_emb = self.graph_pool(output.transpose(0, 1)[mask], src.batch)
        graph_pred = self.task_pred(graph_emb)
        
        return graph_pred


# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math
from torch_geometric.nn import global_mean_pool
from utils import ModifiedAtomEncoder, ModifiedBondEncoder

class RelEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelEncoding, self).__init__()
        self.emb = nn.Embedding(max_len, n_hid)
        self.drop = nn.Dropout(dropout)
        self.emb.weight.data.uniform_(-0.1, 0.1)
    def forward(self, t):
        return self.drop(self.emb(t))

class GT(nn.Module):
    def __init__(self, n_hid, n_out, n_heads, n_layers, edge_dim_dict, dropout = 0.2, summary_node = True):
        super(GT, self).__init__()
        self.node_encoder = ModifiedAtomEncoder(emb_dim=n_hid, summary_node=summary_node)
        self.n_hid     = n_hid
        self.n_out     = n_out
        self.drop      = nn.Dropout(dropout)
        self.gcs       = nn.ModuleList([GT_Layer(n_hid, n_heads, edge_dim_dict, dropout, summary_node)\
                                      for _ in range(n_layers)])
        self.out       = nn.Linear(n_hid, n_out)
        self.summary_node = summary_node

    def forward(self, node_attr, batch_idx, edge_index, strats):
        # strats: edge_attr, cn_edge_attr, sd_edge_attr, etc.
        node_rep = self.node_encoder(node_attr)
        for gc in self.gcs:
            node_rep = gc(node_rep, edge_index, strats)
        if self.summary_node:
            # change to use virtual node
            return self.out(node_rep[node_attr.sum(dim=1) < 0])
        return self.out(global_mean_pool(node_rep, batch_idx))
        

class GT_Layer(MessagePassing):
    def __init__(self, n_hid, n_heads, edge_dim_dict, dropout = 0.2, summary_node = True, **kwargs):
        super(GT_Layer, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.n_hid         = n_hid
        self.n_heads       = n_heads
        self.d_k           = n_hid // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None
        
        
        self.k_linear   = nn.Linear(n_hid,   n_hid)
        self.q_linear   = nn.Linear(n_hid,   n_hid)
        self.v_linear   = nn.Linear(n_hid,   n_hid)
        self.a_linear   = nn.Linear(n_hid,   n_hid)
        self.norm       = nn.LayerNorm(n_hid)
        self.drop       = nn.Dropout(dropout)
        
        self.struc_enc = nn.ModuleDict({
            key : RelEncoding(max_len = edge_dim_dict[key], n_hid = n_hid, dropout = dropout)
                for key in edge_dim_dict if key != 'ea'
        })
        if 'ea' in edge_dim_dict:
            self.struc_enc['ea'] = ModifiedBondEncoder(emb_dim=n_hid, dropout = dropout, summary_node = summary_node)
        
        self.mid_linear  = nn.Linear(n_hid,  n_hid * 2)
        self.out_linear  = nn.Linear(n_hid * 2,  n_hid)
        self.out_norm    = nn.LayerNorm(n_hid)
        self.summary_node = summary_node
        
    def forward(self, node_inp, edge_index, strats):
        return self.propagate(edge_index, node_inp=node_inp, \
                              strats=strats)

    def message(self, edge_index_i, node_inp_i, node_inp_j, strats):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
                
        target_node_vec = node_inp_i
        source_node_vec = node_inp_j 
        for key in self.struc_enc:
            # TODO (low priority): learn different embeddings for different values of k hops
            if self.summary_node:
                if key != 'ea':
                    attr = strats[key]
                    mask = attr >= 0
                    attr_emb = self.struc_enc[key](attr[mask])
                    mod_attr_emb = torch.empty(attr.size(0), attr_emb.size(1), device=attr.get_device())
                    mod_attr_emb[mask] = attr_emb
                    mod_attr_emb[~mask] = 0
                else:
                    source_node_vec += self.struc_enc[key](strats[key])
            else:
                source_node_vec += self.struc_enc[key](strats[key])
                

        q_mat = self.q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = self.k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        v_mat = self.v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = self.drop(softmax((q_mat * k_mat).sum(dim=-1) / self.sqrt_dk, edge_index_i))
        res = v_mat * self.att.view(-1, self.n_heads, 1)
        return res.view(-1, self.n_hid)


    def update(self, aggr_out, node_inp):
        trans_out = self.norm(self.drop(self.a_linear(F.gelu(aggr_out))) + node_inp)
        trans_out = self.out_norm(self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out)
        return trans_out
# -


