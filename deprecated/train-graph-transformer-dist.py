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
import datetime

from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser(description='PyTorch implementation of relative positional encodings and relation-aware self-attention for graph Transformers')
parser.add_argument('-k', type=int, default=2, dest='k_hop_neighbors')
args = parser.parse_args()
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
args.lr = 5e-5
args.graph_pooling = 'mean'
args.proj_mode = 'nonlinear'
args.eval_metric = 'rocauc'
args.embed_dim = 320
args.ff_embed_dim = 640
args.num_heads = 8
args.graph_layers = 4
args.dropout = 0.4
args.relation_type = 'shortest_dist'
args.pre_transform = compute_mutual_shortest_distances
args.max_vocab = 12
args.split = 'scaffold'
args.num_epochs = 200
# args.k_hop_neighbors = 2
args.weights_dropout = True
args.grad_acc = 48
args.cycle_steps = -1
args.warmup_steps = -1
args.weight_decay = 0.01


# %%
def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def train(rank, num_epochs, world_size):
    init_process(rank, world_size)
    
    if rank == 0:
        print("k =", args.k_hop_neighbors)
        print("Loading data...")
        print("dataset: {} ".format(args.dataset))
        dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform)
    dist.barrier()
    print("Loading data...")
    print("dataset: {} ".format(args.dataset))
    dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform)
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
            valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, drop_last=True)
    elif args.split == '80-20':
        train_sampler = DistributedSampler(
            dataset[:int(0.8 * len(dataset))], rank=rank, num_replicas=world_size
        )
        
        train_loader = DataLoader(dataset[:int(0.8 * len(dataset))], batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=train_sampler)
        if rank == 0:
            valid_loader = DataLoader(dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))], batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = GraphTransformerModel(args).cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    batch_device = torch.device('cuda:'+ str(rank) if torch.cuda.is_available() else 'cpu')
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    args.warmup_steps = 1000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader) // args.grad_acc, epochs=num_epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_steps, after_scheduler=scheduler)
    criterion = torch.nn.BCEWithLogitsLoss(reduction = "mean")
    evaluator = Evaluator(name=args.dataset)

    if rank == 0:
        args.writer = SummaryWriter(log_dir=f'runs/molhiv/k={args.k_hop_neighbors}/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    best_valid_score = -1
    best_valid_epoch = -1
    num_iters = 0
    for epoch in range(num_epochs):
        ############
        # TRAINING #
        ############
        train_sampler.set_epoch(epoch + 1)
        
        model.train()

        loss_epoch = 0
        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            batch = batch.to(batch_device, non_blocking=True)
            z = model(batch)

            y = batch.y.float()
            is_valid = ~torch.isnan(y)

            loss = criterion(z[is_valid], y[is_valid]) / args.grad_acc 
            loss.backward()
            # gradient accumulation
            if (idx + 1) % args.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler_warmup.step()
                if rank == 0:
                    args.writer.add_scalar("LR/iters", optimizer.param_groups[0]['lr'], num_iters + 1)
                    num_iters += 1

            loss_epoch += loss.detach().item()
        
        if rank == 0:
            print('Epoch:', epoch + 1)
            args.writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch + 1)
            args.writer.add_scalar("LR/epoch", optimizer.param_groups[0]['lr'], epoch + 1)
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
                for idx, batch in enumerate(valid_loader):
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
        
            args.writer.add_scalar("Loss/valid", loss_epoch / len(valid_loader), epoch + 1)
            print('Valid loss:', loss_epoch / len(valid_loader))
            print('Valid ROC-AUC:', result_dict[args.eval_metric])
            
            print()

            if result_dict[args.eval_metric] >= best_valid_score:
                torch.save(
                    model.state_dict(),
                    f'./models/model_{epoch + 1}_{args.dataset}_lr{args.lr}_k{args.k_hop_neighbors}.pth'
                )
                best_valid_score = result_dict[args.eval_metric]
                best_valid_epoch = epoch + 1
    
    if rank == 0:
        print(f'k={args.k_hop_neighbors}, best valid epoch={best_valid_epoch}, best valid score={best_valid_score}')
        
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,4,5,6'
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        train, args=(args.num_epochs, WORLD_SIZE),
        nprocs=WORLD_SIZE, join=True
    )
    args.writer.close()


# %%
# pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
