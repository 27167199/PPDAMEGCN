import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import itertools
from sklearn import metrics
import dgl

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_pos_neg_ij(adj):
    num_p, num_d = adj.shape
    positive_ij = []
    negative_ij = []
    for i in range(num_p):
        for j in range(num_d):
            label = adj[i, j]
            if label == 1:
                positive_ij.append((i, j))
            elif label == 0:
                negative_ij.append((i, j))
    pos_ij = np.array(positive_ij)
    neg_ij = np.array(negative_ij)
    return pos_ij, neg_ij


def gen_folds(adj):
    pos_ij = np.argwhere(adj == 1)
    neg_ij = np.argwhere(adj == 0)

    positive_idx = np.array(range(0, len(pos_ij)))
    np.random.shuffle(positive_idx)
    negative_idx = np.array(range(0, len(neg_ij)))
    np.random.shuffle(negative_idx)

    pos_5fold_train_idx = []
    pos_5fold_test_idx = []
    neg_5fold_train_idx = []
    neg_5fold_test_idx = []

    kf = KFold(n_splits=5)
    for train, test in kf.split(positive_idx):
        positive_train_idx = positive_idx[train]
        pos_5fold_train_idx.append(positive_train_idx)
        positive_test_idx = positive_idx[test]
        pos_5fold_test_idx.append(positive_test_idx)

    for train, test in kf.split(negative_idx[0 : len(positive_idx)]):
        negative_train_idx = negative_idx[train]
        neg_5fold_train_idx.append(negative_train_idx)
        negative_test_idx = negative_idx[train]
        neg_5fold_test_idx.append(negative_test_idx)

    for i in range(len(pos_5fold_train_idx)):
        train_mask = np.zeros_like(adj, dtype=int)
        test_mask = np.zeros_like(adj, dtype=int)
        train_fold_idx = np.concatenate(
            (pos_ij[pos_5fold_train_idx[i]], neg_ij[neg_5fold_train_idx[i]])
        )
        test_fold_idx = np.concatenate(
            (pos_ij[pos_5fold_test_idx[i]], neg_ij[neg_5fold_test_idx[i]])
        )
        train_mask[tuple(list(train_fold_idx.T))] = 1
        test_mask[tuple(list(test_fold_idx.T))] = 1

        yield train_mask, test_mask


def matrix(a, b, match_score=3, gap_cost=2):
    H = np.zeros((len(a) + 1, len(b) + 1), int)

    for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
        match = H[i - 1, j - 1] + (
            match_score if a[i - 1] == b[j - 1] else -match_score
        )
        delete = H[i - 1, j] - gap_cost
        insert = H[i, j - 1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H


def traceback(H, b, b_="", old_i=0):
    # flip H to get index of **last** occurrence of H.max() with np.argmax()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(
        H.shape, (i_ + 1, j_ + 1)
    )  # (i, j) are **last** indexes of H.max()
    if H[i, j] == 0:
        return b_, j
    b_ = b[j - 1] + "-" + b_ if old_i - i > 1 else b[j - 1] + b_
    return traceback(H[0:i, 0:j], b, b_, i)


def smith_waterman(a, b, match_score=3, gap_cost=2):
    a, b = a.upper(), b.upper()
    H = matrix(a, b, match_score, gap_cost)
    b_, pos = traceback(H, b)
    return pos, pos + len(b_)


class Logger:
    def __init__(self, total_fold):
        def gen_dict():
            return {
                "epoch": [],
                "f1_score": [],
                "f2_score": [],
                "rank_idx": [],
                "auc": [],
                "aupr": [],
                "threshold": [],
                "recall": [],
                "precision": [],
                "acc": [],
                "specificity": [],
                "mcc": [],
                "train_loss": [],
                "test_loss": [],
            }

        self.df = [gen_dict() for i in range(total_fold)]

    def evaluate(self, true, pred_ratings, test_idx, graph_data):
        # rating_values = dataset.test_truths

        rating_values = graph_data['test'][2]

        y_score = pred_ratings.view(-1).cpu().tolist()
        y_true = rating_values.cpu().tolist()
        # auc = metrics.roc_auc_score(y_true, y_score)
        # aupr = metrics.average_precision_score(y_true, y_score)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)

        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr = metrics.auc(recall, precision)

        labels = np.array(y_true)

        scores = np.array(y_score)

        combined = list(zip(labels, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        labels_sorted, scores_sorted = zip(*combined)

        indices = np.arange(1, len(labels) + 1)[np.array(labels_sorted) == 1]
        n_test = len(test_idx)
        n_test_p = sum(labels == 1)
        rank_idx = indices.sum() / n_test / n_test_p

        fpr, tpr, thresholds_ = metrics.roc_curve(labels, scores)
        auc = metrics.auc(fpr, tpr)

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, scores)
        aupr = metrics.auc(recalls, precisions)
        num1 = 2 * recalls * precisions
        den1 = recalls + precisions
        den1[den1 == 0] = 100
        f1_scores = num1 / den1
        f1_score = f1_scores.max()
        beta2 = 2
        num2 = (1 + beta2**2) * recalls * precisions
        den2 = recalls + precisions * beta2**2
        den2[den2 == 0] = 100
        f2_scores = num2 / den2
        f2_score = f2_scores.max()
        f2_score_idx = np.argmax(f2_scores)
        threshold = thresholds[np.argmax(f2_scores)]
        precision = precisions[f2_score_idx]
        recall = recalls[f2_score_idx]
        bi_scores = scores.copy()
        bi_scores[bi_scores < threshold] = 0
        bi_scores[bi_scores >= threshold] = 1
        acc = metrics.accuracy_score(labels, bi_scores)
        tn, fp, fn, tp = metrics.confusion_matrix(labels, bi_scores).ravel()
        specificity = tn / (tn + fp)
        # mcc = metrics.matthews_corrcoef(labels, bi_scores)
        mcc = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return tuple(
            np.round(
                [
                    f1_score,
                    f2_score,
                    rank_idx,
                    auc,
                    aupr,
                    threshold,
                    recall,
                    precision,
                    acc,
                    specificity,
                    mcc,
                ],
                6,
            )
        )

    def update(self, fold, epoch, adj, pred, test_idx, train_loss, test_loss, graph_data):
        (
            f1_score,
            f2_score,
            rank_idx,
            auc,
            aupr,
            threshold,
            recall,
            precision,
            acc,
            specificity,
            mcc,
        ) = self.evaluate(adj, pred, test_idx, graph_data)
        self.df[fold]["epoch"].append(epoch)
        self.df[fold]["f1_score"].append(f1_score)
        self.df[fold]["f2_score"].append(f2_score)
        self.df[fold]["rank_idx"].append(rank_idx)
        self.df[fold]["auc"].append(auc)
        self.df[fold]["aupr"].append(aupr)
        self.df[fold]["threshold"].append(threshold)
        self.df[fold]["recall"].append(recall)
        self.df[fold]["precision"].append(precision)
        self.df[fold]["acc"].append(acc)
        self.df[fold]["specificity"].append(specificity)
        self.df[fold]["mcc"].append(mcc)
        self.df[fold]["train_loss"].append(train_loss)
        self.df[fold]["test_loss"].append(test_loss)
        print(
            f"fold:{fold}, epoch:{epoch}, f1: {f1_score}, f2: {f2_score}, rank_idx: {rank_idx}, auc: {auc}, "
            f"aupr: {aupr}, acc: {acc}, specificity: {specificity}, threshold: {threshold}, recall: {recall}, "
            f"precision: {precision}, mcc: {mcc}, train_loss: {train_loss}, test_loss: {test_loss}"
        )

    def save(self, name):
        with pd.ExcelWriter(f"{name}.xlsx") as writer:
            for fold in range(len(self.df)):
                pd.DataFrame(self.df[fold]).to_excel(
                    writer, sheet_name=f"fold{fold}", index=False
                )

def generate_pair_value(rel_info):
    rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                             dtype=np.int64),
                    np.array([ele for ele in rel_info["disease_id"]],
                             dtype=np.int64))
    rating_values = rel_info["values"].values.astype(np.float32)
    return rating_pairs, rating_values

def generate_topoy_graph(cv_data_dict, args):
    data_cv = {}

    train_data, test_data, values = cv_data_dict
    shuffled_idx = np.random.permutation(train_data.shape[0])
    train_rel_info = train_data.iloc[shuffled_idx[::]]
    test_rel_info = test_data
    possible_rel_values = values
    args.rating_vals = possible_rel_values

    train_pairs, train_values = generate_pair_value(train_rel_info)

    test_pairs, test_values = generate_pair_value(test_rel_info)

    train_enc_graph = generate_enc_graph(train_pairs, train_values, args, values, add_support=True)
    train_dec_graph = generate_dec_graph(train_pairs, args)
    train_truths = torch.FloatTensor(train_values)

    test_enc_graph = train_enc_graph
    test_dec_graph = generate_dec_graph(test_pairs, args)
    test_truths = torch.FloatTensor(test_values)
    data_cv = {'train': [train_enc_graph, train_dec_graph, train_truths],
                        'test': [test_enc_graph, test_dec_graph, test_truths]}
    return data_cv


def generate_enc_graph(rating_pairs, rating_values, args, values, add_support=False):
    symm = args.gcn_agg_norm_symm
    possible_rel_values = values

    data_dict = dict()
    num_nodes_dict = {'drug': args.num_drug, 'disease': args.num_disease}
    rating_row, rating_col = rating_pairs
    for rating in possible_rel_values:
        ridx = np.where(
            rating_values == rating)
        rrow = rating_row[ridx]
        rcol = rating_col[ridx]
        rating = to_etype_name(rating)
        data_dict.update({
            ('drug', str(rating), 'disease'): (rrow, rcol),
            ('disease', 'rev-%s' % str(rating), 'drug'): (rcol, rrow)
        })

    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # sanity check
    assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

    if add_support:
        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = th.FloatTensor(1. / np.sqrt(x))
            return x.unsqueeze(1)

        drug_ci = []
        drug_cj = []
        disease_ci = []
        disease_cj = []
        for r in possible_rel_values:
            r = to_etype_name(r)
            drug_ci.append(graph['rev-%s' % r].in_degrees())
            disease_ci.append(graph[r].in_degrees())
            if symm:
                drug_cj.append(graph[r].out_degrees())
                disease_cj.append(graph['rev-%s' % r].out_degrees())
            else:
                drug_cj.append(th.zeros((args.num_drug,)))
                disease_cj.append(th.zeros((args.num_disease,)))

        drug_ci = _calc_norm(sum(drug_ci))
        disease_ci = _calc_norm(sum(disease_ci))
        if symm:
            drug_cj = _calc_norm(sum(drug_cj))
            disease_cj = _calc_norm(sum(disease_cj))
        else:
            drug_cj = th.ones(args.num_drug, )
            disease_cj = th.ones(args.num_disease, )
        graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
        graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

    return graph


def generate_dec_graph(rating_pairs,args):
    ones = np.ones_like(rating_pairs[0])
    drug_disease_rel_coo = sp.coo_matrix(
        (ones, rating_pairs),
        shape=(args.num_drug, args.num_disease), dtype=np.float32)
    g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                 vtype='_V')
    return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                           num_nodes_dict={'drug': args.num_drug, 'disease': args.num_disease})


import csv
import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim

from scipy import sparse as sp
from collections import OrderedDict


class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) + \
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def common_loss(emb1, emb2):
    emb1 = emb1 - th.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - th.mean(emb2, dim=0, keepdim=True)
    emb1 = th.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = th.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = th.matmul(emb1, emb1.t())
    cov2 = th.matmul(emb2, emb2.t())
    cost = th.mean((cov1 - cov2) ** 2)
    return cost


def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True


# def knn_graph(disMat, k):
#     num  = disMat.shape[0]
#     inds = []
#     for i in range(disMat.shape[0]):
#         ind = np.argpartition(disMat[i, :], kth=k)[:k]
#         inds.append(ind)

#     inds_ = []
#     for i, v in enumerate(inds):
#         for vv in v:
#             if vv == i:
#                 pass
#             else:
#                 inds_.append([i, vv])

#     inds_ = np.array(inds_)
#     edges = np.array([inds_[:, 0], inds_[:,1]]).astype(int)
#     edges_inver = np.array([inds_[:, 1], inds_[:, 0]]).astype(int)
#     edges_index = np.concatenate((edges, edges_inver), axis=1).T
#     # Remove repeating entry
#     edges_index = np.unique(edges_index, axis=0)
#     adjs = sp.coo_matrix((np.ones(edges_index.shape[0]), (edges_index[:, 0], edges_index[:, 1])),
#                                 shape=(num, num), dtype=np.float32)
#     return adjs + sp.eye(adjs.shape[0])


def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(disMat.shape[0], disMat.shape[0]),
                        dtype=np.float32)
    # Remove diagonal elements
    # drug_adj = drug_adj - sp.dia_matrix((drug_adj.diagonal()[np.newaxis, :], [0]), shape=drug_adj.shape)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj

import numpy as np
from sklearn.neighbors import NearestNeighbors

def construct_knn_graph(similarity_matrix, k):
    """
    Constructs the k-nearest neighbor graph adjacency matrix from the similarity matrix.

    Parameters:
    similarity_matrix (numpy.ndarray): The similarity matrix (n x n) where n is the number of drugs.
    k (int): The number of nearest neighbors to consider.

    Returns:
    numpy.ndarray: The k-nearest neighbor adjacency matrix (binary matrix).
    """
    # Number of drugs
    n = similarity_matrix.shape[0]

    # Ensure the diagonal is zeroed out (self-similarity is not considered)
    np.fill_diagonal(similarity_matrix, 0)

    # Use NearestNeighbors to find the k-nearest neighbors based on the similarity matrix
    nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(1 - similarity_matrix)
    distances, indices = nbrs.kneighbors(1 - similarity_matrix)

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Populate the adjacency matrix
    for i in range(n):
        for j in indices[i]:
            adjacency_matrix[i, j] = 1

    return adjacency_matrix+np.eye(n)
    

import random
def data_processing(md_matrix):
    #md_matrix = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(123)
    random.shuffle(one_index)
    random.shuffle(zero_index)



    unsamples = zero_index

    index1 = np.array(one_index, np.int)
    label1 = np.array([1] * len(one_index), dtype=np.int)
    samples1 = np.concatenate((index1, np.expand_dims(label1, axis=1)), axis=1)
    index0 = np.array(zero_index, np.int)
    label0 = np.array([0] * len(zero_index), dtype=np.int)
    samples0 = np.concatenate((index0, np.expand_dims(label0, axis=1)), axis=1)

    return index1, index0

import math
def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])

    # 利用广播和矩阵运算来计算高斯核矩阵
    adj = adj.numpy()
    distances_squared = np.sum((adj[:, np.newaxis, :] - adj) ** 2, axis=-1)  # 计算所有节点对之间的距离的平方
    Gaussian = np.exp(-gama * distances_squared)
    '''
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))
    '''
    return Gaussian

def make_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj