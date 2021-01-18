#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric.transforms as T
from torch import lgamma
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import argparse
import numpy as np
import random
import ogb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from torch.nn import Parameter
import math

from networkx.algorithms.shortest_paths.generic import shortest_path 
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data

from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import os

parser = argparse.ArgumentParser(description='PyTorch implementation of relative positional encodings and relation-aware self-attention for graph Transformers')
args = parser.parse_args("")
args.device = 0
args.device = torch.device('cuda:'+ str(args.device) if torch.cuda.is_available() else 'cpu')
# args.device = torch.device('cpu')
print("device:", args.device)
# torch.cuda.set_device(args.device)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed = 0
set_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %%
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
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
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
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(-1),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)


        attn_weights = F.softmax(attn_weights, dim=1)

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


# %%
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


# %%
args.dataset = 'ogbg-moltox21'
args.n_classes = 12
args.batch_size = 32
args.lr = 0.001
args.graph_pooling = 'mean'
args.proj_mode = 'nonlinear'
args.eval_metric = 'rocauc'
args.embed_dim = 512
args.ff_embed_dim = 1024
args.num_heads = 8
args.graph_layers = 4
args.dropout = 0.2
args.relation_type = 'shortest_dist'
args.pre_transform = compute_mutual_shortest_distances
args.max_vocab = 12
args.split = 'scaffold'
args.num_epochs = 50


# %%
class GraphTransformerModel(nn.Module):

    def __init__(self, nclasses, layers, embed_dim, ff_embed_dim, num_heads, dropout, relation_type, max_vocab, weights_dropout=True):
        super(GraphTransformerModel, self).__init__()
        self.model_type = 'GraphTransformerModel'
        self.encoder = AtomEncoder(emb_dim=embed_dim)
        self.transformer = GraphTransformer(layers, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout)
        
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
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, nclasses)
        )
        
        self.relation_type = relation_type
        self.max_vocab = max_vocab
        self.relation_encoder = nn.Embedding(max_vocab, embed_dim)

    def forward(self, src, src_mask=None):
        x, mask = to_dense_batch(self.encoder(src.x), batch=src.batch, fill_value=0)
        x = x.transpose(0, 1)
        
        if self.relation_type == 'link':
            relation = self.relation_encoder(to_dense_adj(src.edge_index, batch=src.batch, max_num_nodes=x.size(0)).long())
        elif self.relation_type == 'shortest_dist':
            mod_sd_edge_attr = torch.clamp(src.sd_edge_attr.reshape(-1), 0, self.max_vocab - 1).long()
            relation = self.relation_encoder(to_dense_adj(src.sd_edge_index, batch=src.batch, edge_attr=mod_sd_edge_attr, max_num_nodes=x.size(0)).long())
        else:
            raise ValueError("Invalid relation type.")
        
        relation = relation.permute(2, 1, 0, 3)
        
        output = self.transformer(x, relation, self_padding_mask=~mask.transpose(0, 1))
        
        graph_emb = self.graph_pool(output.transpose(0, 1)[mask], src.batch)
        graph_pred = self.task_pred(graph_emb)
        
        return graph_pred


# %%
def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def train(rank, num_epochs, world_size):
    init_process(rank, world_size)
    
    if rank == 0:
        print("Loading data...")
        print("dataset: {} ".format(args.dataset))
        dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform).shuffle()
    dist.barrier()
    print("Loading data...")
    print("dataset: {} ".format(args.dataset))
    dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform).shuffle()
    print(
        f"{rank + 1}/{world_size} process initialized.\n"
    )
    
    split_idx = dataset.get_idx_split()
    sampler = DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=False
    )
    
    if args.split == 'scaffold':
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, pin_memory=True)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, pin_memory=True)
    elif args.split == '80-20':
        train_loader = DataLoader(dataset[:int(0.8 * len(dataset))], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, pin_memory=True)
        test_loader = DataLoader(dataset[int(0.8 * len(dataset)):], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler, pin_memory=True)

    model = GraphTransformerModel(args.n_classes, args.graph_layers, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.relation_type, args.max_vocab).cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    batch_device = torch.device('cuda:'+ str(rank) if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
    evaluator = Evaluator(name=args.dataset)

    for epoch in range(num_epochs):
        ############
        # TRAINING #
        ############

        model.train()

        loss_epoch = 0
        for idx, batch in enumerate(train_loader):
            z = model(batch.to(batch_device, non_blocking=True))

            y = batch.y.float()
            is_valid = ~torch.isnan(y)

            optimizer.zero_grad()
            loss = criterion(z[is_valid], y[is_valid])
            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach().item()

        print('Train loss:', loss_epoch / len(train_loader))

        ##############
        # EVALUATION #
        ##############

        model.eval()

        with torch.no_grad():
            loss_epoch = 0
            y_true = []
            y_scores = []
            for idx, batch in enumerate(test_loader):
                z = model(batch.to(batch_device, non_blocking=True))

                y = batch.y.float()
                y_true.append(y)
                y_scores.append(z)
                is_valid = ~torch.isnan(y)

                optimizer.zero_grad()
                loss = criterion(z[is_valid], y[is_valid])

                loss_epoch += loss.detach().item()

            y_true = torch.cat(y_true, dim = 0)
            y_scores = torch.cat(y_scores, dim = 0)

        input_dict = {"y_true": y_true, "y_pred": y_scores}
        result_dict = evaluator.eval(input_dict)
        print('Test loss:', loss_epoch / len(test_loader))
        print('Test ROC-AUC:', result_dict[args.eval_metric])
        
WORLD_SIZE = torch.cuda.device_count()
if __name__=="__main__":
    mp.spawn(
        train, args=(args.num_epochs, WORLD_SIZE),
        nprocs=WORLD_SIZE, join=True
    )

