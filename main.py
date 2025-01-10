import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#import matplotlib
import pickle
from utils import *
from params import set_args
from data import DrugDataLoader
from evaluate import evaluate

#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cpu")


import os

data_path = "data"

path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)


# load adj, sim
# Load adjacent matrix(data_processing.py)
adj_np = pd.read_csv(f"../{data_path}/adj.csv", index_col=0).values
# Load piRNA similarity based on Smith-Waterman method(gen_half_p2p_simth.py)
p_sim_np = pd.read_csv(f"../{data_path}/p2p_smith.csv", index_col=0).values
# Load piRNA feature based on word2vec method(gen_pfeat_gensim.py)
gensim_feat = np.load(
    f"../{data_path}/gensim_feat_128.npy",
    allow_pickle=True,
).flat[0]
p_kmers_emb = gensim_feat["p_kmers_emb"]
pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]
# Load disease similarity based on DO DAG(gen_d2d_do.py)
d_sim_np = pd.read_csv(f"../{data_path}/d2d_do.csv", index_col=0).values
d_feat = d_sim_np

num_p, num_d = adj_np.shape


p_sim = torch.FloatTensor(p_sim_np).to(device)
d_sim = torch.FloatTensor(d_sim_np).to(device)
adj = torch.FloatTensor(adj_np).to(device)
p_kmers_emb = torch.FloatTensor(p_kmers_emb).to(device)
pad_kmers_id_seq = torch.tensor(pad_kmers_id_seq).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)

args = set_args()
args.num_drug = adj.shape[0]
args.num_disease = adj.shape[1]
args.device = torch.device('cpu')

k = 1
merge_win_size = 32
context_size_list = [1, 3, 5]
dll_out_size = 128 * len(context_size_list) * k

gcn_out_dim = 128 * k
gcn_hidden_dim = 128 * k
num_layers, dropout = 1, 0.4

query_size = key_size = 256 * k
value_size = 256 * k
enc_ffn_num_hiddens, n_enc_heads = 256, 2 * k

lr, num_epochs = 0.001, 150

feat_init_d = d_feat.shape[1]


dataset = DrugDataLoader(args.data_name, args.device,
                         symm=args.gcn_agg_norm_symm,
                         k=args.num_neighbor)

class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        test_loss = (unweighted_loss * test_mask).sum()
        return train_loss, test_loss


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def fit(
    fold_cnt,
    model,
    adj,
    adj_full,
    pad_kmers_id_seq,
    d_feat,
    train_mask,
    test_mask,
    lr,
    num_epochs,
    graph_data,
    drug_graph,
    dis_graph,
    drug_sim_feat,
    dis_sim_feat
):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #loss = MaskedBCELoss()
    rel_loss = nn.BCEWithLogitsLoss()
    #test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    test_idx = torch.nonzero(test_mask.bool()).squeeze()


    #drug_graph = dataset.drug_graph.to(args.device)
    #dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)


    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)
    #drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature

    # rating_values = dataset.test_truths
    rating_values = graph_data['test'][2]
    test_enc_graph = graph_data['test'][0].int().to(args.device)
    test_dec_graph = graph_data['test'][1].int().to(args.device)

    for epoch in range(num_epochs):
        # for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        #pred = model(pad_kmers_id_seq, d_feat, adj_full)


        pred, p_feat_gcn, d_feat_gcn, drug_sim_out, dis_sim_out = model(pad_kmers_id_seq, train_enc_graph, train_dec_graph, drug_graph, dis_graph, drug_sim_feat, dis_sim_feat)

        loss_com_drug = common_loss(p_feat_gcn, drug_sim_out)
        loss_com_dis = common_loss(d_feat_gcn, dis_sim_out)

        #train_loss, test_loss = loss(pred, adj, train_mask, test_mask)

        loss = rel_loss(pred.squeeze(-1), train_gt_ratings) + args.beta * loss_com_dis + args.beta * loss_com_drug
        loss.backward()
        #grad_clipping(model, 1)
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        test_loss= loss
        model.eval()
        pred_ratings,_,_,_,_ = model(pad_kmers_id_seq, test_enc_graph, test_dec_graph, drug_graph, dis_graph, drug_sim_feat, dis_sim_feat)
        '''
        auroc, aupr, y_true, y_score = evaluate(args, model, graph_data,
                                                drug_graph, drug_feat, drug_sim_feat,
                                                dis_graph, dis_feat, dis_sim_feat,pad_kmers_id_seq)
        print("fold:{}, epoch:{}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f}".format(fold_cnt, epoch, loss.item(), auroc, aupr))

        #pred = model(pad_kmers_id_seq, d_feat, adj_full, train_enc_graph, drug_graph, dis_graph, drug_sim_feat,
                     #disease_sim_feat)

        #scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        # print(len(set(scores)))
        '''
        scores = pred_ratings.cpu().detach().numpy().squeeze()
        if(epoch==149):
          #print(scores.shape)
          np.save(rf"./scores/f{fold_cnt}_e{epoch}_scores.npy", scores)
        
        logger.update(
            fold_cnt, epoch, adj, pred_ratings, test_idx, loss.item(), test_loss.item(), graph_data
        )

    return 0


logger = Logger(5)

with open(rf"../{data_path}/fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)
'''
with open(rf"../PUTransGCN_spy/rn_ij_list.pickle", "rb") as f:
    rn_ij_list_spy = pickle.load(f)
with open(rf"../PUTransGCN_pu_bagging/rn_ij_list.pickle", "rb") as f:
    rn_ij_list_pu = pickle.load(f)
with open(rf"../PUTransGCN_two_step/rn_ij_list.pickle", "rb") as f:
    rn_ij_list_two = pickle.load(f)
'''
pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["p_gip_list"]
d_gip_list = fold_info["d_gip_list"]

for i in range(5):
    print(f"fold {i}")
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    p_gip = p_gip_list[i]
    d_gip = d_gip_list[i]

    # rn_ij = rn_ij_list[i]
    #rn_ij = np.concatenate((rn_ij_list_spy[i], rn_ij_list_pu[i], rn_ij_list_two[i]))
    #rn_ij =  np.concatenate((rn_ij_list_pu[i], rn_ij_list_two[i]))
    rn_ij = unlabelled_train_ij
    '''
    train_pos_edge = pos_train_ij.T
    train_pos_values = [1] * len(train_pos_edge[0])
    train_neg_edge = rn_ij.T
    train_neg_values = [0] * len(train_neg_edge[0])

    test_pos_edge = pos_test_ij.T
    test_pos_values = [1] * len(test_pos_edge[0])
    test_neg_edge = unlabelled_test_ij.T
    test_neg_values = [0] * len(test_neg_edge[0])

    train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
    train_values = np.concatenate([train_pos_values, train_neg_values])
    test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
    test_values = np.concatenate([test_pos_values, test_neg_values])

    train_data = {
        'drug_id': train_edge[0],
        'disease_id': train_edge[1],
        'values': train_values
    }
    train_data_info = pd.DataFrame(train_data, index=None)

    test_data = {
        'drug_id': test_edge[0],
        'disease_id': test_edge[1],
        'values': test_values
    }
    test_data_info = pd.DataFrame(test_data, index=None)
    values = np.unique(train_values)
    cv_data = [train_data_info, test_data_info, values]

    graph_data = generate_topoy_graph(cv_data, args)
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    '''
    graph_data = dataset.data_cv[i]


    A_corner_np = np.zeros_like(adj_np)
    A_corner_np[tuple(list(pos_train_ij.T))] = 1

    A_np = np.concatenate(
        (
            np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
            np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
        ),
        axis=0,
    )
    #drug_sim_feat = p_sim_np * np.where(p_sim_np > 0, 1, 0) + p_gip * np.where(p_sim_np > 0, 0, 1)
    #disease_sim_feat = d_sim_np * np.where(d_sim_np > 0, 1, 0) + d_gip * np.where(d_sim_np > 0, 0, 1)
    drug_sim_feat = (p_sim_np + p_gip) / 2
    disease_sim_feat = (d_sim_np + d_gip) / 2
    #drug_sim_feat = p_sim_np
    #disease_sim_feat = d_sim_np
    drug_graph = torch.Tensor(construct_knn_graph(drug_sim_feat, args.num_neighbor)).to(args.device)
    dis_graph = torch.Tensor(construct_knn_graph(disease_sim_feat, args.num_neighbor)).to(args.device)

    drug_sim_feat = torch.FloatTensor(drug_sim_feat).to(args.device)
    disease_sim_feat = torch.FloatTensor(disease_sim_feat).to(args.device)

    # train_mask_np = np.ones_like(adj_np)
    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    # train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1
    train_mask_np[tuple(list(rn_ij.T))] = 1

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1

    A_corner = torch.FloatTensor(A_corner_np).to(device)
    A = torch.FloatTensor(A_np).to(device)
    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)

    torch.cuda.empty_cache()
    '''
    args.src_in_units = A.shape[0]
    args.dst_in_units = A.shape[0]
    args.fdim_drug = 128
    args.fdim_disease = 128
    '''
    #args.src_in_units = 128
    #args.dst_in_units = 128

    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    args.rating_vals = dataset.possible_rel_values
    #args.fdim_drug =128




    deep_lnc_loc = DeepLncLoc(
        p_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
    ).to(device)

    p_encoder = TransformerEncoder(
        q_in_dim=gcn_out_dim*2,
        kv_in_dim=gcn_out_dim*2,
        key_size=key_size,
        query_size=query_size,
        value_size=value_size,
        ffn_num_hiddens=enc_ffn_num_hiddens,
        num_heads=n_enc_heads,
        num_layers=num_layers,
        dropout=dropout,
        bias=False,
    ).to(device)

    d_encoder = TransformerEncoder(
        q_in_dim=gcn_out_dim*2,
        kv_in_dim=gcn_out_dim*2,
        key_size=key_size,
        query_size=query_size,
        value_size=value_size,
        ffn_num_hiddens=enc_ffn_num_hiddens,
        num_heads=n_enc_heads,
        num_layers=num_layers,
        dropout=dropout,
        bias=False,
    ).to(device)

    model = Net(args, deep_lnc_loc)
    model = model.to(args.device)

    fit(
        i,
        model,
        adj,
        A,
        pad_kmers_id_seq,
        d_feat,
        train_mask,
        test_mask,
        lr,
        num_epochs,
        graph_data,
        drug_graph,
        dis_graph,
        drug_sim_feat,
        disease_sim_feat
    )
    max_allocated_memory = torch.cuda.max_memory_allocated()
    print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")

#logger.save(f"PPDAMEGCN_{data_path}")
logger.save("PPDAMEGCN_max")
# torch.save(model, "params.pt")
