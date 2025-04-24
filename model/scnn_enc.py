import math
import pickle
import numpy as np
from numpy.linalg import inv, pinv
import networkx as nx

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter

from model.incidence_matrix import get_faces, incidence_matrices, compute_D1, compute_D2


class SCNNEncoder(object):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(SCNNEncoder, self).__init__()
        self.p = params
        self.device = edge_index.device
        self.drop = nn.Dropout(self.p.dropout)

        # split in/out edge
        num_edges = edge_index.size(1) // 2
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        # boundary matrices
        if self.p.load_h_m:
            L0u = np.load('./HodgeMatrix/' + self.p.dataset + '/' + 'L0u-edge' + str(self.p.edge_sample_scale) + '.npy')
            L1f = np.load('./HodgeMatrix/' + self.p.dataset + '/' + 'L1f-edge' + str(self.p.edge_sample_scale) + '.npy')
            L0u = torch.tensor(L0u, dtype=torch.float32)  # convert hodge matrix to tensor format
            L1f = torch.tensor(L1f, dtype=torch.float32)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.boundary_matrices = [L0u.to(device), L1f.to(device)]
        else:
            self.matrix_operator = MatrixOperator(edge_index, edge_type, num_rel, self.p)
            self.boundary_matrices = self.matrix_operator.get_boundary_matrices()

        boundary_matrix_size = self.boundary_matrices[0].size(0)
        self.weights_L_0 = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), 32)).to(self.device)
        self.weights_L_1 = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), 32)).to(self.device)
        self.weights_off_diagonal = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size), int(boundary_matrix_size))).to(self.device)
        self.in_embeddings_sim = nn.Parameter(torch.FloatTensor(self.p.init_dim, int(boundary_matrix_size * 2))).to(self.device)
        self.in_weights_sim = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size*2), self.p.init_dim)).to(self.device)
        self.out_embeddings_sim = nn.Parameter(torch.FloatTensor(self.p.init_dim, int(boundary_matrix_size * 2))).to(self.device)
        self.out_weights_sim = nn.Parameter(torch.FloatTensor(int(boundary_matrix_size * 2), self.p.init_dim)).to(self.device)

        # incoming and outgoing feature weights
        self.weights_in = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)
        self.weights_out = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)

        # relation weight
        self.w_rel = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)

        # batch norm
        self.bn = torch.nn.BatchNorm1d(self.p.init_dim).to(self.device)

        # reset parameters
        nn.init.kaiming_uniform_(self.weights_off_diagonal, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_L_0, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_L_1, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.in_embeddings_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.in_weights_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_embeddings_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_weights_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_in, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_out, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_rel, mode='fan_out', a=math.sqrt(5))

    def s_encode(self, init_embed, rel_embed):
        L0, L1 = self.boundary_matrices
        sim_block = self.message_passing(L0, L1)

        # in edge
        in_s_emb_sim = self.get_ent_enb(init_embed, rel_embed, sim_block, mode='in')
        # out edge
        out_s_emb_sim = self.get_ent_enb(init_embed, rel_embed, sim_block, mode='out')

        # batch norm
        s_emb_sim = self.drop(in_s_emb_sim) * 1/2 + self.drop(out_s_emb_sim) * 1/2
        s_emb_sim = self.bn(s_emb_sim)

        # relation embedding
        rel_embed = torch.matmul(rel_embed, self.w_rel)

        return s_emb_sim, rel_embed

    def get_ent_enb(self, init_embed, rel_embed, sim_block, mode='in'):
        edge_type = getattr(self, '{}_type'.format(mode))
        edge_index = getattr(self, '{}_index'.format(mode))
        weights = getattr(self, 'weights_{}'.format(mode))
        index = getattr(self, '{}_index'.format(mode))
        embeddings_sim = getattr(self, '{}_embeddings_sim'.format(mode))
        weights_sim = getattr(self, '{}_weights_sim'.format(mode))

        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        tail_ent_emb = init_embed[edge_index[1]]
        xj_rel = self.rel_transform(tail_ent_emb, rel_emb)
        ent_emb = torch.mm(xj_rel, weights)
        ent_emb = self.scatter_(ent_emb, index[0], dim_size=init_embed.size(0))

        embeddings_sim = torch.matmul(ent_emb, embeddings_sim)
        s_emb_sim_ = torch.matmul(embeddings_sim, sim_block)
        s_emb_sim = torch.matmul(s_emb_sim_, weights_sim)
        s_emb_sim = s_emb_sim.renorm_(2, 0, 1)

        return s_emb_sim

    def message_passing(self, L0, L1):
        """
        :param L0: L0 Matrix
        :param L1: L1 Matrix
        :return:
        """
        L0_r = torch.matrix_power(L0, 2)
        L1_r = torch.matrix_power(L1, 2)
        # torch.cuda.empty_cache()

        relation_embedded = torch.einsum('xd, dy -> xy', torch.matmul(L0_r, self.weights_L_0), torch.matmul(L1_r, self.weights_L_1).transpose(0, 1))
        relation_embedded_ = torch.matmul(self.weights_off_diagonal, relation_embedded)
        upper_block = torch.cat([L0_r, relation_embedded_], dim=1)
        lower_block = torch.cat([torch.transpose(relation_embedded_, 0, 1), L1_r], dim=1)
        sim_block = torch.cat([upper_block, lower_block], dim=0)
        sim_block = F.softmax(F.relu(sim_block), dim=1)

        return sim_block

    @staticmethod
    def rel_transform(tail_ent_emb, rel_emb):
        trans_embed = tail_ent_emb - rel_emb
        return trans_embed

    @staticmethod
    def scatter_(src, index, dim_size=None, name='add'):
        if name == 'add':
            name = 'sum'
        assert name in ['sum', 'mean', 'max']
        out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
        return out[0] if isinstance(out, tuple) else out


class MatrixOperator(object):

    def __init__(self, edge_index, edge_type, num_rel, params=None):
        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_rel = num_rel

        if self.p.save_edge_info:
            edge_index = edge_index.cpu().numpy()
            edge_type = edge_type.cpu().numpy()
            data_list = [edge_index, edge_type, num_rel, self.p.num_ent]
            with open('./' + self.p.dataset + '.data', mode="wb") as fw:
                pickle.dump(data_list, fw)

    def get_boundary_matrices(self):
        # random_edge_num = math.floor(self.edge_index.size(1) * self.p.edge_sample_scale)
        random_edge_num = self.p.edge_sample_scale
        indices = np.random.choice(self.edge_index.size(1), (random_edge_num,), replace=False)
        indices = np.sort(indices)
        sample_data_edge_index = self.edge_index[:, indices]

        boundary_matrix0_, boundary_matrix1_ = self.compute_hodge_matrix(sample_data_edge_index)
        L0u, L1f = self.compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        L0u = torch.tensor(L0u, dtype=torch.float32)  # convert hodge matrix to tensor format
        L1f = torch.tensor(L1f, dtype=torch.float32)
        boundary_matrices = [L0u.to(device), L1f.to(device)]
        return boundary_matrices

    def compute_hodge_matrix(self, sample_data_edge_index):
        g = nx.Graph()
        g.add_nodes_from([i for i in range(self.p.num_ent)])
        edge_index_ = np.array(sample_data_edge_index.cpu())
        edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])]
        g.add_edges_from(edge_index)

        edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

        B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

        return B1, B2

    @staticmethod
    def compute_bunch_matrices(B1, B2):
        # D matrices
        D2_2 = compute_D2(B2)
        D2_1 = compute_D2(B1)
        D3_n = np.identity(B1.shape[1])  # (|E| x |E|)
        D1 = compute_D1(B1, D2_2)
        D3 = np.identity(B2.shape[1]) / 3  # (|F| x |F|)

        # L matrices
        D1_pinv = pinv(D1)
        # 对于需要计算伪逆矩阵的大型矩阵，可以使用numpy的lstsq函数（最小二乘函数）来代替pinv函数，从而获得更快的速度。
        # D1_pinv = np.linalg.lstsq(D1, np.identity(D1.shape[0]), rcond=None)[0]
        D2_2_inv = inv(D2_2)

        L0u = B1.T @ B1  # B1 @ D3_n @ B1.T @ inv(D2_1)
        L1u = D2_2 @ B1.T @ D1_pinv @ B1
        L1d = B2 @ D3 @ B2.T @ D2_2_inv
        L1f = L1u + L1d

        return L0u, L1f
