import pandas as pd
import os
import torch
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import linalg
from .data import MyDataLoader


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# class MyScaler:
#     def __init__(self, mean, std):
#         self.mean = 0
#         self.std = std
#
#     def transform(self, data):
#         return (data - self.mean) / self.std
#
#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean
#
# def explained_variance(pred, y):
#     return 1 - torch.var(y - pred) / torch.var(y)
#
#
# def Myloss(type, inputs, targets):
#     # print(f'input: {inputs.shape}, targets: {targets.shape}')
#     if type == "mse":
#         return F.mse_loss(inputs, targets)
#     raise NameError("Loss not supported:", type)

def load_features(feat_path, dtype=np.float32, dataset=None, time=False):
    if dataset == 'loop':
        feat_df = pd.read_pickle('./../data/speed_matrix_2015')
    else:
        feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    # print(f'Loading feature...\n feat shape: {feat.shape}')

    if dataset == 'hk' and time:
        num_data, num_sensor = feat.shape
        feat = np.expand_dims(feat, axis=-1)
        feat_hk = feat.tolist()

        for i in range(num_data):
            time = (i % 288) / 288
            for j in range(num_sensor):
                feat_hk[i][j].append(time)

        feat_hk = np.array(feat_hk)

        return feat_hk

    if dataset == 'loop' and time:
        num_data, num_sensor = feat.shape
        feat = np.expand_dims(feat, axis=-1)
        feat_loop = feat.tolist()

        for i in range(num_data):
            time = (i % 288) / 288
            for j in range(num_sensor):
                feat_loop[i][j].append(time)

        feat_loop = np.array(feat_loop)

        return feat_loop

    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32, dataset=None):
    if dataset == 'loop':
        adj_df = np.load('./../data/Loop_Seattle_2015_A.npy')
    else:
        adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    # print(f'Loading adj...\n adj shape: {adj.shape}')

    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, labelrate=100, divison_seed=0
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    # print(f'feat shape: {data.shape}')

#    print(f'37472: {data[37472]}')

    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    # scaler = StandardScaler(mean=data.mean(), std=data.std())
    # print(f'mean: {scaler.mean}, std: {scaler.std}')

    max_val = np.max(data)
    # print(f'max_val: {max_val}')
    if time_len is None:
        time_len = data.shape[0]
    # if normalize:
        # data = data / max_val
        # data = scaler.transform(data)
    train_size = int(time_len * split_ratio)
    val_size = int(time_len * (1-split_ratio) / 3)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:time_len]

    print(f'SHAPE | train:{train_data.shape} | val:{val_data.shape} | test:{test_data.shape}')

    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
    # print(f'Num train: {len(train_data)}, Num val: {len(val_data)}, Num test: {len(test_data)}')

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(np.array(val_data[i : i + seq_len]))
        val_Y.append(np.array(val_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)


    if labelrate != 100:
        import random
        random.seed(divison_seed)
        # sample = [i for i in range(train_X.shape[0])]
        # train_sample = random.sample(sample, int(train_X.shape[0] * labelrate / 100))
        # train_X = train_X[train_sample]
        # train_Y = train_Y[train_sample]

        # sample = [i for i in range(train_X.shape[0]-(int(train_X.shape[0] * labelrate / 100) + 1))]
        train_sample = random.randint(0, (train_X.shape[0]-(int(train_X.shape[0] * labelrate / 100) + 1)))
        print(f'train_samle: {train_sample}')
        train_X = train_X[train_sample: train_sample + (int(train_X.shape[0] * labelrate / 100))]
        train_Y = train_Y[train_sample: train_sample + (int(train_Y.shape[0] * labelrate / 100))]

    print(f'\ntrain_X: {train_X.shape}')
    print(f'train_Y: {train_Y.shape}')
    print(f'val_X: {val_X.shape}')
    print(f'val_Y: {val_Y.shape}')
    print(f'test_X: {test_X.shape}')
    print(f'test_Y: {test_Y.shape}\n')

    max_xtrain = np.max(train_X)
    max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    min_xtrain = np.min(train_X)
    min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
    min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

    # print(f'max_speed: {max_speed}, min_speed: {min_speed}')

    scaler = StandardScaler(mean=train_X[..., 0].mean(), std=train_X[..., 0].std())

    print(f'max_speed: {max_speed}, min_speed: {min_speed}, avg: {scaler.mean}, std: {scaler.std}')

    # print(f'scaler: {scaler.mean}, {scaler.std}')

    if normalize:
        train_X = scaler.transform(train_X)
        train_Y = scaler.transform(train_Y)
        val_X = scaler.transform(val_X)
        val_Y = scaler.transform(val_Y)
        test_X = scaler.transform(test_X)
        test_Y = scaler.transform(test_Y)

    max_xtrain = np.max(train_X)
    max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    min_xtrain = np.min(train_X)
    min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
    min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

    print(f'normalized max_speed: {max_speed}, normalized min_speed: {min_speed}')

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler

#
# def generate_torch_datasets(
#     data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
# ):
#     train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = generate_dataset(
#         data,
#         seq_len,
#         pre_len,
#         time_len=time_len,
#         split_ratio=split_ratio,
#         normalize=normalize,
#     )
#
#     print(f'\ntrain_X: {train_X.shape}')
#     print(f'train_Y: {train_Y.shape}')
#     print(f'val_X: {val_X.shape}')
#     print(f'val_Y: {val_Y.shape}')
#     print(f'test_X: {test_X.shape}')
#     print(f'test_Y: {test_Y.shape}\n')
#
#     # train_dataset = torch.utils.data.TensorDataset(
#     #     torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
#     # )
#     # val_dataset = torch.utils.data.TensorDataset(
#     #     torch.FloatTensor(val_X), torch.FloatTensor(val_Y)
#     # )
#     # test_dataset = torch.utils.data.TensorDataset(
#     #     torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
#     # )
#
#     train_dataset = MyTensorDataset(
#         torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
#     )
#     val_dataset = MyTensorDataset(
#         torch.FloatTensor(val_X), torch.FloatTensor(val_Y)
#     )
#     test_dataset = MyTensorDataset(
#         torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
#     )
#
#     return train_dataset, val_dataset, test_dataset, max_speed, scaler

def load_all_adj(device):
    adj_loop = load_adjacency_matrix("./../data/Loop_Seattle_2015_A.npy", dataset='loop')
    _, _, adj_la = load_adj('./../data/adj_mx.pkl', 'doubletransition')
    _, _, adj_bay = load_adj('./../data/adj_mx_bay.pkl', 'doubletransition')
    adj_pems04 = get_adjacency_matrix(distance_df_filename="./../data/PEMS04/PEMS04.csv", num_of_vertices=307)
    adj_pems07 = get_adjacency_matrix(distance_df_filename="./../data/PEMS07/PEMS07.csv", num_of_vertices=883)
    adj_pems08 = get_adjacency_matrix(distance_df_filename="./../data/PEMS08/PEMS08.csv", num_of_vertices=170)

    return torch.tensor(adj_loop).to(device), torch.tensor(adj_la).to(device), torch.tensor(adj_bay).to(device), \
           torch.tensor(adj_pems04).to(device), torch.tensor(adj_pems07).to(device), torch.tensor(adj_pems08).to(device)


def load_data(args, scaler=None, visualize=False, distribution=False):
    DATA_PATHS = {
        "shenzhen": {"feat": "./../data/sz_speed.csv", "adj": "./../data/sz_adj.csv"},
        "losloop": {"feat": "./../sata/los_speed.csv", "adj": "./../data/los_adj.csv"},
        "hongkong": {"feat": "./../data/HongKong/HKdata.csv", "adj": "./../data/HongKong/HKadj.csv"},
        "Seattle": {"feat": "./../data/speed_matrix_2015", "adj": "./../data/Loop_Seattle_2015_A.npy"},
        "3": {"feat": "./../data/PEMS03/PEMS03.npz", "adj": "./../data/PEMS03/PEMS03.csv"},
        "4": {"feat": "./../data/PEMS04/PEMS04.npz", "adj": "./../data/PEMS04/PEMS04.csv"},
        "7": {"feat": "./../data/PEMS07/PEMS07.npz", "adj": "./../data/PEMS07/PEMS07.csv"},
        "8": {"feat": "./../data/PEMS08/PEMS08.npz", "adj": "./../data/PEMS08/PEMS08.csv"},
    }


    time = False
    if args.model in ['DCRNN', 'STGCN']:
        time = True

    if args.dataset == 'sz':
        featpath = DATA_PATHS['shenzhen']['feat']
        adjpath = DATA_PATHS['shenzhen']['adj']
        feat = load_features(featpath)
        adj = load_adjacency_matrix(adjpath)

    elif args.dataset == 'los':
        featpath = DATA_PATHS['losloop']['feat']
        adjpath = DATA_PATHS['losloop']['adj']
        feat = load_features(featpath)
        adj = load_adjacency_matrix(adjpath)

    elif args.dataset == 'hk':
        featpath = DATA_PATHS['hongkong']['feat']
        adjpath = DATA_PATHS['hongkong']['adj']
        feat = load_features(featpath, dataset='hk', time=time)
        adj = load_adjacency_matrix(adjpath)

    elif args.dataset == 'loop':
        featpath = DATA_PATHS['Seattle']['feat']
        adjpath = DATA_PATHS['Seattle']['adj']
        feat = load_features(featpath, dataset='loop', time=time)
        adj = load_adjacency_matrix(adjpath, dataset='loop')

    elif args.dataset == '3':
        feat_dir = DATA_PATHS['3']['feat']
        adj_dir = DATA_PATHS['3']['adj']
        num_of_vertices = 358

    elif args.dataset == '4':
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

    if distribution:
        data = load_distribution(feat_dir)
        return data
    else:
        if args.dataset == 'la':
            sensor_ids, sensor_id_to_ind, adj_mx = load_adj('./../data/adj_mx.pkl', 'doubletransition')
            dataloader = load_dataset('./../data/METR-LA', args.batch_size, args.labelrate, time, args.division_seed)

        elif args.dataset == 'bay':
            sensor_ids, sensor_id_to_ind, adj_mx = load_adj('./../data/adj_mx_bay.pkl', 'doubletransition')
            dataloader = load_dataset('./../data/PEMS-BAY', args.batch_size, args.labelrate, time, args.division_seed)

        if args.dataset in ['sz', 'los', 'loop', 'hk']:
            train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = generate_dataset(
                feat,
                args.seq_len,
                args.pre_len,
                time_len=None,
                split_ratio=args.split_ratio,
                normalize=args.normalize,
                labelrate=args.labelrate,
                divison_seed=args.division_seed,
            )
            train_dataloader = MyDataLoader(torch.FloatTensor(train_X), torch.FloatTensor(train_Y), batch_size=args.batch_size)
            val_dataloader = MyDataLoader(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), batch_size=args.batch_size)
            test_dataloader = MyDataLoader(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), batch_size=args.batch_size)
        elif args.dataset in ['3', '4', '7', '8']:
            train_X, train_Y, val_X, val_Y, test_X, test_Y, max_speed, scaler = load_graphdata_channel1(args, feat_dir, time, scaler, visualize=visualize)
            train_dataloader = MyDataLoader(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                            batch_size=args.batch_size)
            val_dataloader = MyDataLoader(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), batch_size=args.batch_size)
            test_dataloader = MyDataLoader(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), batch_size=args.batch_size)
            adj = get_adjacency_matrix(distance_df_filename=adj_dir, num_of_vertices=num_of_vertices)
            # print(f'adj shape: {adj.shape}, num of 1s: {adj.sum()}\n{adj}')
        else:
            train_dataloader = dataloader["train_loader"]
            val_dataloader = dataloader["val_loader"]
            test_dataloader = dataloader["test_loader"]
            scaler = dataloader["scaler"]
            max_speed = dataloader["max"]
            adj = adj_mx

        return train_dataloader, val_dataloader, test_dataloader, torch.tensor(adj), max_speed, scaler

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx.cpu())
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    # print(f'adj_ori: {np.array(adj_mx)}')
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"

    # print(f'adj_mx: {adj_mx}')
    return sensor_ids, sensor_id_to_ind, adj_mx

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def load_distribution(feat_dir):
    train_X = train_Y = val_X = val_Y = test_X = test_Y = max_val = None

    file_data = np.load(feat_dir)
    # for k in file_data.files:
    #     print(k)

    data = file_data['data']
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0
    data = data[:, :, 0]  # flow only
    data = np.array(data)

    print(f'shape: {data.shape}')

    return data

def load_graphdata_channel1(args, feat_dir, time, scaler=None, visualize=False):
    """
        dir: ./../data/PEMS03/PEMS03.npz, shape: (26208, 358, 1) 91 days, 2018, 9.1 - 11.30, [flow]                 15%
        dir: ./../data/PEMS04/PEMS04.npz, shape: (16992, 307, 3) 59 days, 2018, 1.1 - 2.28 , [flow, occupy, speed]  24%
        dir: ./../data/PEMS07/PEMS07.npz, shape: (28224, 883, 1) 98 days, 2017, 5.1 - 8.31 , [flow]                 14%
        dir: ./../data/PEMS08/PEMS08.npz, shape: (17856, 170, 3) 62 days, 2016, 7.1 - 8.31 , [flow, occupy, speed]  23%
    """
    train_X = train_Y = val_X = val_Y = test_X = test_Y = max_val = None


    file_data = np.load(feat_dir)
    # for k in file_data.files:
    #     print(k)

    data = file_data['data']
    where_are_nans = np.isnan(data)
    data[where_are_nans] = 0
    where_are_nans = (data != data)
    data[where_are_nans] = 0
    data = data[:, :, 0]  # flow only

    if time:
        num_data, num_sensor = data.shape
        data = np.expand_dims(data, axis=-1)
        data = data.tolist()

        for i in range(num_data):
            time = (i % 288) / 288
            for j in range(num_sensor):
                data[i][j].append(time)

        data = np.array(data)

    # print(f'data shape: {data.shape}')
    max_val = np.max(data)
    time_len = data.shape[0]
    seq_len = args.seq_len
    pre_len = args.pre_len
    split_ratio = args.split_ratio
    train_size = int(time_len * split_ratio)
    val_size = int(time_len * (1 - split_ratio) / 3)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:time_len]

    if args.labelrate != 100:
        import random
        new_train_size = int(train_size * args.labelrate / 100)
        start = random.randint(0, train_size - new_train_size - 1)
        train_data = train_data[start:start+new_train_size]

    print(f'SHAPE | train:{train_data.shape} | val:{val_data.shape} | test:{test_data.shape}')

    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
    # print(f'Num train: {len(train_data)}, Num val: {len(val_data)}, Num test: {len(test_data)}')

    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i: i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(np.array(val_data[i: i + seq_len]))
        val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i: i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

    if visualize:
        test_X = test_X[-288:]
        test_Y = test_Y[-288:]

    if args.labelrate != 0:
        train_X = np.array(train_X)
        train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)



    if args.labelrate != 0:
        max_xtrain = np.max(train_X)
        max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    if args.labelrate != 0:
        min_xtrain = np.min(train_X)
        min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    if args.labelrate != 0:
        max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

        scaler = StandardScaler(mean=train_X[..., 0].mean(), std=train_X[..., 0].std())

        print(f'max_flow: {max_speed}, min_flow: {min_speed}, avg: {scaler.mean}, std: {scaler.std}')

        train_X = scaler.transform(train_X)
        train_Y = scaler.transform(train_Y)
    else:
        max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

        print(f'max_flow: {max_speed}, min_flow: {min_speed}, avg: {scaler.mean}, std: {scaler.std}')

    val_X = scaler.transform(val_X)
    val_Y = scaler.transform(val_Y)
    test_X = scaler.transform(test_X)
    test_Y = scaler.transform(test_Y)

    if args.labelrate != 0:
        max_xtrain = np.max(train_X)
        max_ytrain = np.max(train_Y)
    max_xval = np.max(val_X)
    max_yval = np.max(val_Y)
    max_xtest = np.max(test_X)
    max_ytest = np.max(test_Y)

    if args.labelrate != 0:
        min_xtrain = np.min(train_X)
        min_ytrain = np.min(train_Y)
    min_xval = np.min(val_X)
    min_yval = np.min(val_Y)
    min_xtest = np.min(test_X)
    min_ytest = np.min(test_Y)

    if args.labelrate != 0:
        max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

        print(f'normalized max_flow: {max_speed}, normalized min_flow: {min_speed}')

    else:
        max_speed = max(max_xval, max_yval, max_xtest, max_ytest)
        min_speed = min(min_xval, min_yval, min_xtest, min_ytest)

        print(f'normalized max_flow: {max_speed}, normalized min_flow: {min_speed}')

    if args.labelrate != 0:
        print(f'\ntrain_X: {train_X.shape}')
        print(f'train_Y: {train_Y.shape}')
    print(f'val_X: {val_X.shape}')
    print(f'val_Y: {val_Y.shape}')
    print(f'test_X: {test_X.shape}')
    print(f'test_Y: {test_Y.shape}\n')

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, max_val, scaler

def load_dataset(dataset_dir, batch_size, labelrate, time, transform=True, division_seed=0):
    from .data import MyDataLoader
    import random
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # print(f'x_train: {data["x_train"].shape}')
    # print(f'x_val: {data["x_val"].shape}')

    if labelrate != 100:
        random.seed(division_seed)
        # sample = [i for i in range(data["x_train"].shape[0])]
        # train_sample = random.sample(sample, int(data["x_train"].shape[0] * labelrate /100))
        # print(f'train_sample: {train_sample}')
        # print(int(labelrate*data["x_train"].shape[0]/100))
        # train_sample = random.randint(0, (train_X.shape[0] - (int(train_X.shape[0] * labelrate / 100) + 1)))

        # sample = [i for i in range(data["x_train"].shape[0] - (int(data["x_train"].shape[0] * labelrate /100) + 1))]
        train_sample = random.randint(0, data["x_train"].shape[0] - (int(data["x_train"].shape[0] * labelrate / 100) + 1))
        data["x_train"] = data["x_train"][train_sample: train_sample + int(data["x_train"].shape[0] * labelrate / 100)]
        data["y_train"] = data["y_train"][train_sample: train_sample + int(data["y_train"].shape[0] * labelrate / 100)]

    if not time:
        data["x_train"] = np.array(data['x_train'][..., 0])
        data["y_train"] = np.array(data['y_train'][..., 0])
        data["x_val"] = np.array(data['x_val'][..., 0])
        data["y_val"] = np.array(data['y_val'][..., 0])
        data["x_test"] = np.array(data['x_test'][..., 0])
        data["y_test"] = np.array(data['y_test'][..., 0])

    print(f'x_train: {data["x_train"].shape}')
    print(f'x_val: {data["x_val"].shape}')
    print(f'x_test: {data["x_test"].shape}')
    print(f'y_train: {data["y_train"].shape}')
    print(f'y_val: {data["y_val"].shape}')
    print(f'y_test: {data["y_test"].shape}')

    # print(f'{data["x_train"].shape}, {data["y_train"].shape}, {data["x_val"].shape}, {data["y_val"].shape}, {data["x_test"].shape}, {data["y_test"].shape}')

    max_xtrain = np.max(data['x_train'])
    max_ytrain = np.max(data['y_train'])
    max_xval = np.max(data['x_val'])
    max_yval = np.max(data['y_val'])
    max_xtest = np.max(data['x_test'])
    max_ytest = np.max(data['y_test'])

    min_xtrain = np.min(data['x_train'])
    min_ytrain = np.min(data['y_train'])
    min_xval = np.min(data['x_val'])
    min_yval = np.min(data['y_val'])
    min_xtest = np.min(data['x_test'])
    min_ytest = np.min(data['y_test'])

    max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
    min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # scaler = MyScaler(mean=0, std=max_speed)
    # print(f'scaler: {scaler.mean}, {scaler.std}')

    print(f'max_speed: {max_speed}, min_speed: {min_speed}, avg: {scaler.mean}, std: {scaler.std}')

    for category in ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']:
        data[category] = scaler.transform(data[category])
            # data[category] /= max_speed
    max_xtrain = np.max(data['x_train'])
    max_ytrain = np.max(data['y_train'])
    max_xval = np.max(data['x_val'])
    max_yval = np.max(data['y_val'])
    max_xtest = np.max(data['x_test'])
    max_ytest = np.max(data['y_test'])

    min_xtrain = np.min(data['x_train'])
    min_ytrain = np.min(data['y_train'])
    min_xval = np.min(data['x_val'])
    min_yval = np.min(data['y_val'])
    min_xtest = np.min(data['x_test'])
    min_ytest = np.min(data['y_test'])

    max_speed = max(max_xtrain, max_ytrain, max_xval, max_yval, max_xtest, max_ytest)
    min_speed = min(min_xtrain, min_ytrain, min_xval, min_yval, min_xtest, min_ytest)

    print(f'normalized max_speed: {max_speed}, normalized min_speed: {min_speed}')

    data['train_loader'] = MyDataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = MyDataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = MyDataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler
    data['max'] = max_speed

    return data

def masked_mae_loss(y_pred, y_true):
    # print(f'pred: {y_pred.shape}, true: {y_true.shape}')
    # print(f'pred: {y_pred}, label: {y_true}')

    mask_true = (y_true > 0.01).float()
    mask_pred = (y_pred > 0.01).float()
    mask = torch.mul(mask_true, mask_pred)
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true):
    mask_true = (y_true > 0.01).float()
    mask_pred = (y_pred > 0.01).float()
    mask = torch.mul(mask_true, mask_pred)
    mask /= mask.mean()
    loss = torch.square(y_pred -y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mape_loss(y_pred, y_true):
    mask_true = (y_true > 0.01).float()
    mask_pred = (y_pred > 0.01).float()
    mask = torch.mul(mask_true, mask_pred)
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true) / y_true
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

def masked_loss(y_pred, y_true):
    # mask_true_nan = ~(y_true != y_true)
    # mask_pred_nan = ~(y_pred != y_pred)
    mask_true = (y_true > 0.01).float()
    mask_pred = (y_pred > 0.01).float()
    # mask_true = (mask_true | mask_true_nan).float()
    # mask_pred = (mask_pred | mask_pred_nan).float()
    mask = torch.mul(mask_true, mask_pred)
    mask /= mask.mean()
    mae_loss = torch.abs(y_pred - y_true)
    mse_loss = torch.square(y_pred - y_true)
    mape_loss = mae_loss / y_true
    mae_loss = mae_loss * mask
    mse_loss = mse_loss * mask
    mape_loss = mape_loss * mask
    mae_loss[mae_loss != mae_loss] = 0
    mse_loss[mse_loss != mse_loss] = 0
    mape_loss[mape_loss != mape_loss] = 0

    return mae_loss.mean(), torch.sqrt(mse_loss.mean()), mape_loss.mean()

# def masked_mse(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels!=null_val)
#     mask = mask.float()
#     mask /= torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = (preds-labels)**2
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
#
# def masked_rmse(preds, labels, null_val=np.nan):
#     return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
#
#
# def masked_mae(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels!=null_val)
#     mask = mask.float()
#     mask /=  torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds-labels)
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
#
#
# def masked_mape(preds, labels, null_val=np.nan):
#     if np.isnan(null_val):
#         mask = ~torch.isnan(labels)
#     else:
#         mask = (labels!=null_val)
#     mask = mask.float()
#     mask /=  torch.mean((mask))
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = torch.abs(preds-labels)/labels
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)
#
#
# def metric(pred, real):
#     mae = masked_mae(pred, real, 0.0).item()
#     mape = masked_mape(pred, real, 0.0).item()
#     rmse = masked_rmse(pred, real, 0.0).item()
#     return mae, mape, rmse


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A.numpy(), axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_avg_std(result):
    avg = np.mean(result)
    std = np.std(result)

    return avg, std