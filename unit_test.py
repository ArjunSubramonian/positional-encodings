import torchimport torch.nn as nnfrom torch_geometric.data import DataLoaderfrom tensorboardX import SummaryWriterimport argparseimport numpy as npimport randomimport ogbfrom ogb.graphproppred import PygGraphPropPredDataset, Evaluatorfrom graph_transformer import GraphTransformerModelfrom utils import compute_mutual_shortest_distances, compute_all_node_connectivity, compute_edge_betweenness_centrality, \                    compute_clique_number, type_of_encodingimport torch.multiprocessing as mpimport torch.distributed as distfrom torch.utils.data.distributed import DistributedSamplerfrom torch.nn.parallel import DistributedDataParallel, DataParallelimport osimport datetimeparser = argparse.ArgumentParser(description='PyTorch implementation of relative positional encodings '                                             'and relation-aware self-attention for graph Transformers')args = parser.parse_args("")# %%args.dataset = 'ogbg-molhiv'args.n_classes = 1args.batch_size = 2args.lr = 5e-5args.graph_pooling = 'mean'args.proj_mode = 'nonlinear'args.eval_metric = 'rocauc'args.embed_dim = 320args.ff_embed_dim = 640args.num_heads = 8args.graph_layers = 4args.dropout = 0.4args.relation_type = "connectivity"args.pre_transform = compute_all_node_connectivityargs.max_vocab = 6args.split = 'scaffold'args.num_epochs = 200args.k_hop_neighbors = 4args.weights_dropout = Trueargs.grad_acc = 48args.cycle_steps = -1args.warmup_steps = -1args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")if __name__ == "__main__":    # dist.barrier()    print("Loading data...")    print("dataset: {} ".format(args.dataset))    dataset = PygGraphPropPredDataset(name=args.dataset, pre_transform=args.pre_transform)    split_idx = dataset.get_idx_split()    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,                              shuffle=False, drop_last=True)    model = GraphTransformerModel(args).cuda()    model = DataParallel(model)    batch_device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')    best_valid_score = -1    num_iters = 0    model.train()    print("place2")    loss_epoch = 0    for idx, batch in enumerate(train_loader):        print("I got here")        batch = batch.to(batch_device, non_blocking=True)        print("batch {}".format(batch))        z = model(batch)        break