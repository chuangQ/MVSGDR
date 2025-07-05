import pandas as pd
import argparse
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import networkx as nx
import dgl
import torch.nn.functional as F
import metis
import scipy.sparse as sp
import scipy.io as sio
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    adj = adj.numpy()
    return adj

def k_fold(data, args):
    k = args.k_fold
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    X_train_all, X_train_p_all, X_test_all, X_test_p_all, Y_train_all, Y_test_all = [], [], [], [], [], []
    X_train_n_all, X_test_n_all = [], []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_p = X_train[Y_train[:, 0] == 1, :]
        X_train_n = X_train[Y_train[:, 0] == 0, :]
        X_test_p = X_test[Y_test[:, 0] == 1, :]
        X_test_n = X_test[Y_test[:, 0] == 0, :]
        X_train_all.append(X_train)
        X_train_p_all.append(X_train_p)
        X_train_n_all.append(X_train_n)
        X_test_all.append(X_test)
        X_test_p_all.append(X_test_p)
        X_test_n_all.append(X_test_n)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    data['X_train'] = X_train_all
    data['X_train_p'] = X_train_p_all
    data['X_train_n'] = X_train_n_all
    data['X_test'] = X_test_all
    data['X_test_p'] = X_test_p_all
    data['X_test_n'] = X_test_n_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)
    didi_matrix = k_matrix(data['dis'], args.neighbor)
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)
    didi_nx = nx.from_numpy_matrix(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)
    didi_graph = dgl.from_networkx(didi_nx)

    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])

    return drdr_graph, didi_graph, data


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_data2(data, num):
    num_nodes = num - 1
    # node_feat = data['X_train']
    edge_index = torch.tensor(data).t()

    dataset = {'edge_index': edge_index,
               'node_feat': None,
               'edge_feat': None,
               'num_nodes': num_nodes}

    return dataset


def partition_patch(data, n_patches, load_path=None):
    if load_path is not None:
        patch = torch.load(load_path)
    else:
        if n_patches == 1:
            patch = torch.tensor(range(data['num_nodes'] + 1)).unsqueeze(dim=0)
        else:
            patch = metis_partition(g=data, n_patches=n_patches)
        print('metis done!!!')
    print('patch done!!!')
    data['num_nodes'] += 1
    return patch


def metis_partition(g, n_patches=50):
    if g['num_nodes'] < n_patches:
        membership = torch.randperm(n_patches)
    else:
        # data augmentation
        adjlist = g['edge_index'].t()
        G = nx.Graph()
        G.add_nodes_from(np.arange(g['num_nodes']))
        G.add_edges_from(adjlist.tolist())
        # metis partition
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    assert len(membership) >= g['num_nodes']
    membership = torch.tensor(membership[:g['num_nodes']])

    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [g['num_nodes']] * (max_patch_size - l)

    patch = torch.tensor(patch)

    return patch


def get_data_new(args):
    sdata = dict()
    if args.dataset in ['Gdataset', 'Cdataset']:
        data = sio.loadmat(args.data_dir)
        association_matrix = data['didr'].T
        disease_sim_features = data['disease']
        drug_sim_features = data['drug']
    elif args.dataset in ['Ldataset']:
        association_matrix = np.loadtxt(os.path.join(args.data_dir, 'lagcn/drug_dis.csv'), delimiter=",")
        disease_sim_features = np.loadtxt(os.path.join(args.data_dir, 'lagcn/dis_sim.csv'), delimiter=",")
        drug_sim_features = np.loadtxt(os.path.join(args.data_dir, 'lagcn/drug_sim.csv'), delimiter=",")
    elif args.dataset in ['lrssl']:
        data = pd.read_csv(os.path.join(args.data_dir, 'drug_dis.txt'), index_col=0, delimiter='\t')
        association_matrix = data.values
        disease_sim_features = pd.read_csv(
            os.path.join(args.data_dir, 'dis_sim.txt'), index_col=0, delimiter='\t').values
        drug_sim_features = pd.read_csv(
            os.path.join(args.data_dir, 'drug_sim.txt'), index_col=0, delimiter='\t').values

    sdata['drug_number'] = int(drug_sim_features.shape[0])
    sdata['disease_number'] = int(disease_sim_features.shape[0])

    sdata['drs'] = drug_sim_features
    sdata['dis'] = disease_sim_features
    edges = []
    # 遍历邻接矩阵
    for i in range(association_matrix.shape[0]):  # 遍历行
        for j in range(association_matrix.shape[1]):  # 遍历列
            if association_matrix[i, j] != 0:  # 如果存在边
                edges.append([i, j])  # 添加边到列表中
    sdata['drdi'] = np.array(edges)
    sdata['didr'] = sdata['drdi'][:, [1, 0]]
    return sdata, association_matrix


def data_processing_new(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]

    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    return data

