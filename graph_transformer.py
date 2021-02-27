import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch
from utils import type_of_encoding

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
        self.bin = args.bin

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
            # print("src.batch {}".format(src.batch))
            relation = self.relation_encoder(to_dense_adj(src.sd_edge_index, batch=src.batch, edge_attr=mod_sd_edge_attr,
                                                          max_num_nodes=x.size(0)).long())
        elif self.relation_type == "connectivity":
            other_edge_attr = torch.clamp(src.other_edge_attr.reshape(-1), 0, self.max_feature - 1).long()  # just to ensure
            relation = self.relation_encoder(
                to_dense_adj(src.other_edge_index.long(), batch=src.batch, edge_attr=other_edge_attr.long(),
                             max_num_nodes=x.size(0)).long())
        ## continous
        elif self.relation_type in type_of_encoding:
            ### change according to the encodings ###
            value_max = src.jaccard_max + 2
            bin_size = value_max/self.max_vocab
            other_edge_index = src.jaccard_index.long()
            other_edge_attr = torch.clamp(src.jaccard_attr.reshape(-1), 0, value_max - 1)
            other_edge_attr = (other_edge_attr / bin_size).int().long()
            ### change according to the encodings ###
            relation = self.relation_encoder(
                            to_dense_adj(other_edge_index,
                            batch=src.batch, edge_attr=other_edge_attr,
                            max_num_nodes=x.size(0)).long())
        else:
            raise ValueError("Invalid relation type.")
        
        # integrate given edge features

        mod_edge_attr = self.edge_encoder(src.edge_attr)
        # print("mod_edge_attr {}".format(mod_edge_attr))
        relation += to_dense_adj(src.edge_index, batch=src.batch, edge_attr=mod_edge_attr, max_num_nodes=x.size(0))
        relation = relation.permute(2, 1, 0, 3)
        
        output = self.transformer(x, relation, self_padding_mask=~mask.transpose(0, 1), self_attn_mask=self_attn_mask)
        
        graph_emb = self.graph_pool(output.transpose(0, 1)[mask], src.batch)
        graph_pred = self.task_pred(graph_emb)
        
        return graph_pred
