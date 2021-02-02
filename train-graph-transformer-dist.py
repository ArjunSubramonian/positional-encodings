#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import random
import ogb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from graph_transformer import GraphTransformerModel
from utils import compute_mutual_shortest_distances

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
args.dataset = 'ogbg-molhiv'
args.n_classes = 1
args.batch_size = 2
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
args.weights_dropout = True
args.grad_acc = 256
args.writer = SummaryWriter(log_dir='runs/molhiv')


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
        dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform)
    dist.barrier()
    print("Loading data...")
    print("dataset: {} ".format(args.dataset))
    dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform).shuffle()
    print(
        f"{rank + 1}/{world_size} process initialized.\n"
    )
    
    if args.split == 'scaffold':
        split_idx = dataset.get_idx_split()
        train_sampler = DistributedSampler(
            dataset[split_idx["train"]], rank=rank, num_replicas=world_size
        )
        
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
        if rank == 0:
            test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, drop_last=True)
    elif args.split == '80-20':
        train_sampler = DistributedSampler(
            dataset[:int(0.8 * len(dataset))], rank=rank, num_replicas=world_size
        )
        
        train_loader = DataLoader(dataset[:int(0.8 * len(dataset))], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
        if rank == 0:
            test_loader = DataLoader(dataset[int(0.8 * len(dataset)):], batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = GraphTransformerModel(args).cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    batch_device = torch.device('cuda:'+ str(rank) if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
    criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
    evaluator = Evaluator(name=args.dataset)

    for epoch in range(num_epochs):
        ############
        # TRAINING #
        ############
        train_sampler.set_epoch(epoch)
        
        model.train()

        loss_epoch = 0
        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            # print(rank, idx, '/', len(train_loader), batch)
            batch = batch.to(batch_device, non_blocking=True)
            z = model(batch)

            y = batch.y.float()
            is_valid = ~torch.isnan(y)

            loss = criterion(z[is_valid], y[is_valid]) # / args.grad_acc 
            loss.backward()
            # gradient accumulation
            if (idx + 1) % args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            loss_epoch += loss.detach().item()
        
        if rank == 0:
            print('Epoch:', epoch + 1)
            args.writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch + 1)
        print('Train loss (cuda:' + str(rank) + '):', loss_epoch / len(train_loader))

        if rank == 0:
            ##############
            # EVALUATION #
            ##############

            model.eval()

            with torch.no_grad():
                loss_epoch = 0
                y_true = []
                y_scores = []
                for idx, batch in enumerate(test_loader):
                    batch = batch.to(batch_device, non_blocking=True)
                    z = model(batch)

                    y = batch.y.float()
                    y_true.append(y)
                    y_scores.append(z)
                    is_valid = ~torch.isnan(y)

                    loss = criterion(z[is_valid], y[is_valid])
                    loss_epoch += loss.detach().item()

                y_true = torch.cat(y_true, dim = 0)
                y_scores = torch.cat(y_scores, dim = 0)

            input_dict = {"y_true": y_true, "y_pred": y_scores}
            result_dict = evaluator.eval(input_dict)
        
            args.writer.add_scalar("Loss/test", loss_epoch / len(test_loader), epoch + 1)
            print('Test loss:', loss_epoch / len(test_loader))
            print('Test ROC-AUC:', result_dict[args.eval_metric])

            print()

            if (epoch + 1) % 10 == 0:
                torch.save(
                    model.state_dict(),
                    f'./models/model_{epoch + 1}_{args.dataset}.pth'
                )
        
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='2,3,5,6,7'
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        train, args=(args.num_epochs, WORLD_SIZE),
        nprocs=WORLD_SIZE, join=True
    )
    args.writer.close()


# %%
# try 'bace' dataset
# Longformer, if fully-connected data is too much information
# random sampling for training dataset
# plot learning curve
# warmup steps for learning
# there might be bugs
# double-check DistributedSampler

# %%
