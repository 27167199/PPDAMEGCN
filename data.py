import os
import dgl
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import pickle
from utils import *
from sklearn.model_selection import KFold


_paths = {
    'Gdataset': './raw_data/drug_data/Gdataset/Gdataset.mat',
    'Cdataset': './raw_data/drug_data/Cdataset/Cdataset.mat',
    'Ldataset': './raw_data/drug_data/Ldataset/lagcn',
    'lrssl': './raw_data/drug_data/lrssl',

}


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
   # adj = sp.coo_matrix(adj)
   # adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum = (row_sum == 0)*1 + row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


class DrugNovoLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        self.row_idx = [ith for ith in range(0, self._num_drug)]

        cv_num = 0
        cv_data = {}
        for idx in self.row_idx:
            train_matrix = association_matrix.copy()
            test_value = train_matrix[idx, :]
            test_data = {
                'drug_id': [idx] * len(test_value),
                'disease_id': [col for col in range(0, self._num_disease)],
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            train_matrix[idx, :] = 0
            pos_row, pos_col = np.nonzero(train_matrix)
            neg_row, neg_col = np.nonzero(1 - train_matrix)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_id': train_drug_idx,
                'disease_id': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.row_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_adj = knn_graph(drug_sim, drug_num_neighbor)
        # drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        # dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        # dr_col_index = drug_neighbor.reshape(-1)
        # drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        # drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        # drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
        #                          shape=(self.num_drug, self.num_drug),
        #                          dtype=np.float32)
        # drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
        #     drug_adj.T > drug_adj)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)

        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]
        disease_adj = knn_graph(disease_sim, disease_num_neighbor)

        # disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        # di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        # di_col_index = disease_neighbor.reshape(-1)
        # disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        # disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        # disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
        #                             shape=(self.num_disease, self.num_disease),
        #                             dtype=np.float32)
        # disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
        #     disease_adj.T > disease_adj)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
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
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug
data_path = "data"

class DrugDataLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        #self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data( self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        # self.possible_rel_values = self.values

    def _load_drug_data(self,  data_name):
        association_matrix = None



        if data_name in ['one']:
            association_matrix = pd.read_csv(f"../{data_path}/adj.csv", index_col=0).values
            self.disease_sim_features = pd.read_csv(f"../{data_path}/d2d_do.csv", index_col=0).values
            self.drug_sim_features = pd.read_csv(f"../{data_path}/p2p_smith.csv", index_col=0).values

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]


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
        '''
        adj_np = pd.read_csv("../data/adj.csv", index_col=0).values
        samples1, unsamples= data_processing(adj_np)
        from sklearn.model_selection import KFold
        from utils import make_adj, get_gaussian
        kf = KFold(n_splits= 10, shuffle=True, random_state=10)
        kf_samples1 = [train_test for train_test in kf.split(samples1)]
        kf_samples0 = [train_test for train_test in kf.split(unsamples)]


        cv_num = 0
        cv_data = {}
        for i in range(0, 10):
        
            pos_train_ij = samples1[kf_samples1[i][0]]
            pos_test_ij = samples1[kf_samples1[i][1]]
            unlabelled_train_ij = unsamples[kf_samples0[i][0]]
            unlabelled_test_ij = unsamples[kf_samples0[i][0]]
            
            train_adj = make_adj(samples1[kf_samples1[i][0]], ((adj_np.shape[0]), adj_np.shape[1]))
            p_gip = get_gaussian(train_adj)
            d_gip = get_gaussian(train_adj.T)
            
            rn_ij =  unlabelled_train_ij
            
            #rn_ij = np.concatenate((rn_ij_list_pu[i], rn_ij_list_two[i]))
            #rn_ij =  unlabelled_train_ij_list[i]
            

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
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data
        #for 5cv
        '''
        cv_num = 0
        cv_data = {}
        for i in range(0, 5):
        
            pos_train_ij = pos_train_ij_list[i]
            pos_test_ij = pos_test_ij_list[i]
            unlabelled_train_ij = unlabelled_train_ij_list[i]
            unlabelled_test_ij = unlabelled_test_ij_list[i]
            p_gip = p_gip_list[i]
            d_gip = d_gip_list[i]

            
            rn_ij =  unlabelled_train_ij
            
            #rn_ij = np.concatenate((rn_ij_list_pu[i], rn_ij_list_two[i]))
            #rn_ij =  unlabelled_train_ij_list[i]
            

            train_pos_edge = pos_train_ij_list[i].T
            train_pos_values = [1] * len(train_pos_edge[0])
            train_neg_edge = rn_ij.T
            train_neg_values = [0] * len(train_neg_edge[0])

            test_pos_edge = pos_test_ij_list[i].T
            test_pos_values = [1] * len(test_pos_edge[0])
            test_neg_edge = unlabelled_test_ij_list[i].T
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
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data
    
    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, 5):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_adj = knn_graph(drug_sim, drug_num_neighbor)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_adj = knn_graph(disease_sim, disease_num_neighbor)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph
        
    # def _generate_feat_graph(self):
    #     # drug feature graph
    #     drug_sim = self.drug_sim_features
    #     drug_num_neighbor = self.num_neighbor
    #     if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
    #         drug_num_neighbor = drug_sim.shape[0]

    #     drug_adj = knn_graph(drug_sim, drug_num_neighbor)
    #     drug_graph = sys_normalized_adjacency(drug_adj + sp.eye(drug_adj.shape[0]))
    #     drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        
    #     # disease feature graph
    #     disease_sim = self.disease_sim_features
    #     disease_num_neighbor = self.num_neighbor
    #     if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
    #         disease_num_neighbor = disease_sim.shape[0]

    #     disease_adj = knn_graph(disease_sim, disease_num_neighbor)
    #     disease_graph = sys_normalized_adjacency(disease_adj + sp.eye(disease_adj.shape[0]))
    #     disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

    #     return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
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
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug


class DrugModeloader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._dir, self._name)

        self._generate_topoy_graph()
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()
        self.possible_rel_values = self.values

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Gdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['Ldataset']:
            association_matrix = np.loadtxt(os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]
        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'), index_col=0, delimiter='\t').values
            # self.row_idx = [ele for ele in range(0, association_matrix.shape[0])]

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]
        self.col_idx = [ele for ele in range(0, association_matrix.shape[1])]

        cv_num = 0
        cv_data = {}
        for col in self.col_idx:
            train_matrix = association_matrix.copy()
            test_value = train_matrix[:, col]
            test_data = {
                'drug_id': [idx for idx in range(0, self._num_drug)],
                'disease_id': [col] * len(test_value),
                'values': test_value
            }
            test_data_info = pd.DataFrame(test_data, index=None)

            # train_matrix[:, col] = 0
            pos_row, pos_col = np.nonzero(train_matrix)
            neg_row, neg_col = np.nonzero(1 - train_matrix)

            train_drug_idx = np.hstack([pos_row, neg_row])
            train_disease_idx = np.hstack([pos_col, neg_col])

            pos_values = [1] * len(pos_row)
            neg_values = [0] * len(neg_row)
            train_values = np.hstack([pos_values, neg_values])

            train_data = {
                'drug_id': train_drug_idx,
                'disease_id': train_disease_idx,
                'values': train_values
            }
            train_data_info = pd.DataFrame(train_data, index=None)

            values = np.unique(train_values)
            cv_data[cv_num] = [train_data_info, test_data_info, values]
            cv_num += 1

        return cv_data

    def _generate_feat(self):
        self.drug_feature_shape = (self.num_drug, self.num_drug + self.num_disease + 3)
        self.disease_feature_shape = (self.num_disease, self.num_drug + self.num_disease + 3)

        self.drug_feature = th.cat(
            [th.Tensor(list(range(3, self.num_drug + 3))).reshape(-1, 1), th.zeros([self.num_drug, 1]) + 1,
             th.zeros([self.num_drug, 1])], 1)

        self.disease_feature = th.cat(
            [th.Tensor(list(range(self.num_drug + 3, self.num_drug + self.num_disease + 3))).reshape(-1, 1),
             th.ones([self.num_disease, 1]) + 1, th.zeros([self.num_disease, 1])], 1)

    def _generate_topoy_graph(self):
        self.data_cv = {}
        for cv in range(0, len(self.col_idx)):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]
            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values

            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)

            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)

            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values,
                                                            add_support=True)
            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)

            self.test_enc_graph = self.train_enc_graph
            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)
            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths]}
        return self.data_cv

    def _generate_feat_graph(self):
        # drug feature graph
        drug_sim = self.drug_sim_features
        drug_num_neighbor = self.num_neighbor
        if drug_num_neighbor > drug_sim.shape[0] or drug_num_neighbor < 0:
            drug_num_neighbor = drug_sim.shape[0]

        drug_neighbor = np.argpartition(-drug_sim, kth=drug_num_neighbor, axis=1)[:, :drug_num_neighbor]
        dr_row_index = np.arange(drug_neighbor.shape[0]).repeat(drug_neighbor.shape[1])
        dr_col_index = drug_neighbor.reshape(-1)
        drug_edge_index = np.array([dr_row_index, dr_col_index]).astype(int).T

        drug_edges = np.array(list(drug_edge_index), dtype=np.int32).reshape(drug_edge_index.shape)
        drug_adj = sp.coo_matrix((np.ones(drug_edges.shape[0]), (drug_edges[:, 0], drug_edges[:, 1])),
                                 shape=(self.num_drug, self.num_drug),
                                 dtype=np.float32)
        drug_adj = drug_adj + drug_adj.T.multiply(drug_adj.T > drug_adj) - drug_adj.multiply(
            drug_adj.T > drug_adj)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)
        # disease feature graph
        disease_sim = self.disease_sim_features
        disease_num_neighbor = self.num_neighbor
        if disease_num_neighbor > disease_sim.shape[0] or disease_num_neighbor < 0:
            disease_num_neighbor = disease_sim.shape[0]

        disease_neighbor = np.argpartition(-disease_sim, kth=disease_num_neighbor, axis=1)[:, :disease_num_neighbor]
        di_row_index = np.arange(disease_neighbor.shape[0]).repeat(disease_neighbor.shape[1])
        di_col_index = disease_neighbor.reshape(-1)
        disease_edge_index = np.array([di_row_index, di_col_index]).astype(int).T

        disease_edges = np.array(list(disease_edge_index), dtype=np.int32).reshape(disease_edge_index.shape)
        disease_adj = sp.coo_matrix((np.ones(disease_edges.shape[0]), (disease_edges[:, 0], disease_edges[:, 1])),
                                    shape=(self.num_disease, self.num_disease),
                                    dtype=np.float32)
        disease_adj = disease_adj + disease_adj.T.multiply(disease_adj.T > disease_adj) - disease_adj.multiply(
            disease_adj.T > disease_adj)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph

    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        data_dict = dict()
        num_nodes_dict = {'drug': self._num_drug, 'disease': self._num_disease}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rel_values:
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
            for r in self.possible_rel_values:
                r = to_etype_name(r)
                drug_ci.append(graph['rev-%s' % r].in_degrees())
                disease_ci.append(graph[r].in_degrees())
                if self._symm:
                    drug_cj.append(graph[r].out_degrees())
                    disease_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    drug_cj.append(th.zeros((self.num_drug,)))
                    disease_cj.append(th.zeros((self.num_disease,)))

            drug_ci = _calc_norm(sum(drug_ci))
            disease_ci = _calc_norm(sum(disease_ci))
            if self._symm:
                drug_cj = _calc_norm(sum(drug_cj))
                disease_cj = _calc_norm(sum(disease_cj))
            else:
                drug_cj = th.ones(self.num_drug, )
                disease_cj = th.ones(self.num_disease, )
            graph.nodes['drug'].data.update({'ci': drug_ci, 'cj': drug_cj})
            graph.nodes['disease'].data.update({'ci': disease_ci, 'cj': disease_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_drug, self.num_disease), dtype=np.float32)
        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='_U', etype='_E',
                                     vtype='_V')
        return dgl.heterograph({('drug', 'rate', 'disease'): g.edges()},
                               num_nodes_dict={'drug': self.num_drug, 'disease': self.num_disease})

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug


if __name__ == '__main__':
    # DrugDataLoader("Gdataset", device=th.device('cpu'), symm=True)
    DrugNovoLoader("Gdataset", device=th.device('cpu'), symm=True)
