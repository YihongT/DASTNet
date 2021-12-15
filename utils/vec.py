import networkx as nx
import numpy as np
import pandas as pd
import torch
from multiprocessing import cpu_count
from .funcs import load_adjacency_matrix, get_adjacency_matrix
from .funcs import load_adj

def generate_vector(args):
    DATA_PATHS = {
        "3": {"feat": "./../data/PEMS03/PEMS03.npz", "adj": "./../data/PEMS03/PEMS03.csv"},
        "4": {"feat": "./../data/PEMS04/PEMS04.npz", "adj": "./../data/PEMS04/PEMS04.csv"},
        "7": {"feat": "./../data/PEMS07/PEMS07.npz", "adj": "./../data/PEMS07/PEMS07.csv"},
        "8": {"feat": "./../data/PEMS08/PEMS08.npz", "adj": "./../data/PEMS08/PEMS08.csv"},
    }

    if args.dataset == '4':
        feat_dir = DATA_PATHS['4']['feat']
        adj_dir = DATA_PATHS['4']['adj']
        num_of_vertices = 307

    elif args.dataset == '7':
        feat_dir = DATA_PATHS['7']['feat']
        adj_dir = DATA_PATHS['7']['adj']
        num_of_vertices = 883

    elif args.dataset == '8':
        feat_dir = DATA_PATHS['8']['feat']
        adj_dir = DATA_PATHS['8']['adj']
        num_of_vertices = 170

    if args.dataset in ['4', '7', '8']:
        adj = get_adjacency_matrix(distance_df_filename=adj_dir, num_of_vertices=num_of_vertices)

    graph = nx.DiGraph(adj)
    spl = dict(nx.all_pairs_shortest_path_length(graph))

    # node2vec = Node2Vec(graph, dimensions=args.vec_dim, walk_length=args.walk_length, num_walks=args.num_walks, \
    #                     p=args.p, q=args.q, workers=(int(cpu_count()/2)))
    # n2v = node2vec.fit(window=10, min_count=1, batch_words=4)

    from .node2vec import Node2Vec as node2vec
    n2v = node2vec(G=graph, distance=spl, emb_size=args.vec_dim, length_walk=args.walk_length,
                   num_walks=args.num_walks, window_size=5, batch=4, p=args.p, q=args.q,
                   workers=(int(cpu_count() / 2)))


    n2v = n2v.train()

    if args.testmodify:
        gfeat = []
        for i in range(len(adj)):
            nodeivec = n2v.wv.get_vector(str(i))
            gfeat.append(nodeivec)

        g = []
        for j in range(args.batch_size):
            g.append(gfeat)

        g = torch.tensor(np.array(g))

        # (batch_size, num_nodes, emb_size)
        print(f'n2v shape after broadcast: {g.shape}')
    else:
        gfeat = []
        for i in range(len(adj)):
            nodeivec = n2v.wv.get_vector(str(i))
            gfeat.append(nodeivec)
        g = torch.tensor(np.array(gfeat))
        print(f'n2v shape: {g.shape}')

    return g, gfeat