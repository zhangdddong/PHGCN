import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):

    def __init__(self, edge_index, edge_type, num_rel, params):
        super(GCNEncoder, self).__init__()
        self.p = params
        self.device = edge_index.device
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_rel = num_rel
        self.drop = nn.Dropout(self.p.dropout)

        # split in/out edge
        num_edges = edge_index.size(1) // 2
        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        # incoming and outgoing feature weights
        self.weights_in = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)
        self.weights_out = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)

        # relation weight
        self.w_rel = nn.Parameter(torch.FloatTensor(self.p.init_dim, self.p.init_dim)).to(self.device)

        # GCN encoder
        self.conv1 = GCNConv(self.p.init_dim, self.p.init_dim, cached=True)

        # batch norm
        self.bn = torch.nn.BatchNorm1d(self.p.init_dim).to(self.device)
        self.bn_gcn = torch.nn.BatchNorm1d(self.p.init_dim).to(self.device)

        nn.init.kaiming_uniform_(self.weights_in, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_out, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_rel, mode='fan_out', a=math.sqrt(5))

    def g_encode(self, x, init_rel):

        # aggregate relation features
        # in edge
        in_x = self.get_ent_enb(x, init_rel, mode='in')
        # out edge
        out_x = self.get_ent_enb(x, init_rel, mode='out')
        # batch norm
        x = self.drop(in_x) * 1 / 2 + self.drop(out_x) * 1 / 2
        x = self.bn(x)
        x = F.relu(self.conv1(x, self.edge_index))
        x = self.bn_gcn(x)

        rel_embed = torch.matmul(init_rel, self.w_rel)

        return x, rel_embed

    def get_ent_enb(self, init_embed, rel_embed, mode='in'):
        edge_type = getattr(self, '{}_type'.format(mode))
        edge_index = getattr(self, '{}_index'.format(mode))
        weights = getattr(self, 'weights_{}'.format(mode))
        index = getattr(self, '{}_index'.format(mode))

        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        tail_ent_emb = init_embed[edge_index[1]]
        xj_rel = self.rel_transform(tail_ent_emb, rel_emb)
        ent_emb = torch.mm(xj_rel, weights)
        ent_emb = self.scatter_(ent_emb, index[0], dim_size=init_embed.size(0))

        return ent_emb

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
