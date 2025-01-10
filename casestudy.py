import numpy as np
from utils import *
import torch
from model import *
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# import matplotlib
import pickle
from utils import *
from params import set_args
from data import DrugDataLoader
from evaluate import evaluate

# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

seed_everything(42)
device = torch.device("cpu")

import os

path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)

# load adj, sim
# Load adjacent matrix(data_processing.py)
adj_np = pd.read_csv("../../comparison/data/adj.csv", index_col=0).values
adj1 = pd.read_csv("../../comparison/data/adj.csv", index_col=0)
# Load piRNA similarity based on Smith-Waterman method(gen_half_p2p_simth.py)
p_sim_np = pd.read_csv("../../comparison/data/p2p_smith.csv", index_col=0).values
# Load piRNA feature based on word2vec method(gen_pfeat_gensim.py)
gensim_feat = np.load(
    "../../comparison/data/gensim_feat_128.npy",
    allow_pickle=True,
).flat[0]
p_kmers_emb = gensim_feat["p_kmers_emb"]
pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]
# Load disease similarity based on DO DAG(gen_d2d_do.py)
d_sim_np = pd.read_csv("../../comparison/data/d2d_do.csv", index_col=0).values
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
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def fit(
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
    fold_cnt = 0
    model.apply(xavier_init_weights)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr, 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss = MaskedBCELoss()
    rel_loss = nn.BCEWithLogitsLoss()
    # test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    test_idx = torch.nonzero(test_mask.bool()).squeeze()

    # drug_graph = dataset.drug_graph.to(args.device)
    # dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)

    train_gt_ratings = graph_data['train'][2].to(args.device)
    train_enc_graph = graph_data['train'][0].int().to(args.device)
    train_dec_graph = graph_data['train'][1].int().to(args.device)
    # drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature

    # rating_values = dataset.test_truths
    rating_values = graph_data['test'][2]
    test_enc_graph = graph_data['test'][0].int().to(args.device)
    test_dec_graph = graph_data['test'][1].int().to(args.device)

    for epoch in range(num_epochs):
        # for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        # pred = model(pad_kmers_id_seq, d_feat, adj_full)

        pred, p_feat_gcn, d_feat_gcn, drug_sim_out, dis_sim_out = model(pad_kmers_id_seq, train_enc_graph,
                                                                        train_dec_graph, drug_graph, dis_graph,
                                                                        drug_sim_feat, dis_sim_feat)

        loss_com_drug = common_loss(p_feat_gcn, drug_sim_out)
        loss_com_dis = common_loss(d_feat_gcn, dis_sim_out)

        # train_loss, test_loss = loss(pred, adj, train_mask, test_mask)

        loss = rel_loss(pred.squeeze(-1), train_gt_ratings) + args.beta * loss_com_dis + args.beta * loss_com_drug
        loss.backward()
        # grad_clipping(model, 1)
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        test_loss = loss
        model.eval()
        pred_ratings, _, _, _, _ = model(pad_kmers_id_seq, test_enc_graph, test_dec_graph, drug_graph, dis_graph,
                                         drug_sim_feat, dis_sim_feat)

        scores = pred_ratings.cpu().detach().numpy()
        #logger.update(
            #fold_cnt, epoch, adj, pred_ratings, test_idx, loss.item(), test_loss.item(), graph_data
        #)

    return scores
logger = Logger(1)


pos_ij = np.argwhere(adj_np == 1)
unlabelled_ij = np.argwhere(adj_np == 0)

unlabelled_train_ij = unlabelled_ij[:len(pos_ij)]
unlabelled_ij = unlabelled_ij[len(pos_ij):]

p_gip = get_gaussian(adj_np)
d_gip = get_gaussian(adj_np.T)

graph_data = dataset.data_cv

A_corner_np = np.zeros_like(adj_np)
A_corner_np[tuple(list(pos_ij.T))] = 1

A_np = np.concatenate(
    (
        np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
        np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
    ),
    axis=0,
)
# drug_sim_feat = p_sim_np * np.where(p_sim_np > 0, 1, 0) + p_gip * np.where(p_sim_np > 0, 0, 1)
# disease_sim_feat = d_sim_np * np.where(d_sim_np > 0, 1, 0) + d_gip * np.where(d_sim_np > 0, 0, 1)
drug_sim_feat = (p_sim_np + p_gip) / 2
disease_sim_feat = (d_sim_np + d_gip) / 2
# drug_sim_feat = p_sim_np
# disease_sim_feat = d_sim_np
drug_graph = torch.Tensor(construct_knn_graph(drug_sim_feat, args.num_neighbor)).to(args.device)
dis_graph = torch.Tensor(construct_knn_graph(disease_sim_feat, args.num_neighbor)).to(args.device)

drug_sim_feat = torch.FloatTensor(drug_sim_feat).to(args.device)
disease_sim_feat = torch.FloatTensor(disease_sim_feat).to(args.device)

# train_mask_np = np.ones_like(adj_np)
train_mask_np = np.zeros_like(adj_np)
train_mask_np[tuple(list(pos_ij.T))] = 1
train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1
#train_mask_np[tuple(list(rn_ij.T))] = 1

test_mask_np = np.zeros_like(adj_np)
#test_mask_np[tuple(list(pos_test_ij.T))] = 1
test_mask_np[tuple(list(unlabelled_ij.T))] = 1

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
# args.src_in_units = 128
# args.dst_in_units = 128

args.src_in_units = dataset.drug_feature_shape[1]
args.dst_in_units = dataset.disease_feature_shape[1]
args.fdim_drug = dataset.drug_feature_shape[0]
args.fdim_disease = dataset.disease_feature_shape[0]
args.rating_vals = dataset.possible_rel_values
# args.fdim_drug =128

deep_lnc_loc = DeepLncLoc(
    p_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
).to(device)

p_encoder = TransformerEncoder(
    q_in_dim=gcn_out_dim * 2,
    kv_in_dim=gcn_out_dim * 2,
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
    q_in_dim=gcn_out_dim * 2,
    kv_in_dim=gcn_out_dim * 2,
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

scores = fit(
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
piRNA_list = np.array(adj1.index)
disease_list = np.array(adj1.columns)
disease_piRNA_pair = unlabelled_ij
pair_scores = scores

print(piRNA_list.shape)
print(disease_list.shape)
print(scores.shape)
# 将这些列组合成一个 DataFrame
df = pd.DataFrame({
    'piRNA': piRNA_list[disease_piRNA_pair[:,0]],
    'disease': disease_list[disease_piRNA_pair[:,1]],
    'score': pair_scores.squeeze()
})

#df_sort = df.sort_values(by='score', ascending=False)

df= df.sort_values(by='score', ascending=False)
output_file = 'CaseStudy2_2.csv'
df.to_csv(output_file, index=False)

# 创建一个 Pandas Excel writer
excel_writer = pd.ExcelWriter('CaseStudy2_1.xlsx')

for i in range(disease_list.shape[0]):
    df_disease = df[df['disease'] == disease_list[i]]
    df_disease_sort = df_disease.sort_values(by='score', ascending=False)
    df_disease_sort.to_excel(excel_writer, sheet_name=disease_list[i], index=False)

# 保存 Excel 文件
excel_writer.close()

max_allocated_memory = torch.cuda.max_memory_allocated()
print(f"最大已分配内存量: {max_allocated_memory / 1024 ** 2} MB")



#logger.save("PPDAMEGCN_10CV")
# torch.save(model, "params.pt")
