import torch
import torch.nn.functional as F
from torch.nn import Parameter

from model.scnn_enc import SCNNEncoder
from model.gcn_enc import GCNEncoder
from model.lagcn_conv import LaGCNConv
from model.saf_layer import SAF
from helper import get_param


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class SCKGEBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(SCKGEBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_rel = num_rel

        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.p.score_func == 'transe':
            init_rel = get_param([self.p.num_rel, self.p.init_dim])
            self.init_rel = torch.cat([init_rel, -init_rel], dim=0).to(device)
        else:
            self.init_rel = get_param([self.p.num_rel * 2, self.p.init_dim]).to(device)

        if self.p.gcn_type == 'normal':
            # GCN encoder
            self.gcn_encoder = GCNEncoder(edge_index, edge_type, num_rel, params)
        elif self.p.gcn_type == 'lagcn':
            # laGCN
            self.gcn_encoder = LaGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)

        # Simplicial Complex encoder
        if self.p.use_scnn:
            self.sc_encoder = SCNNEncoder(edge_index, edge_type, num_rel, params)
            # agg
            self.linear_entity = torch.nn.Linear(self.p.init_dim * 2, self.p.init_dim, bias=True)
            self.bn_entity = torch.nn.BatchNorm1d(self.p.init_dim)
            self.linear_rel = torch.nn.Linear(self.p.init_dim * 2, self.p.init_dim, bias=True)
            self.bn_rel = torch.nn.BatchNorm1d(self.p.init_dim)

        # SAF
        if self.p.agg_method == 'att':
            self.saf = SAF(self.p.init_dim, self.p.init_dim)

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2):

        # Graph Laplacian operation
        if self.p.gcn_type == 'normal':
            x_g_emb, r_g_embed = self.gcn_encoder.g_encode(self.init_embed, self.init_rel)
        elif self.p.gcn_type == 'lagcn':
            x_g_emb, r_g_embed = self.gcn_encoder(self.init_embed, self.edge_index, self.edge_type, rel_embed=self.init_rel)

        if self.p.use_scnn:
            # Hodge Laplacian operation
            x_s_emb, r_s_embed = self.sc_encoder.s_encode(self.init_embed, self.init_rel)

            if self.p.agg_method == 'att':
                x_embed = self.saf(x_g_emb, x_s_emb)
                r_embed = self.bn_rel(self.linear_rel(torch.cat((self.p.xi * r_g_embed, self.p.mu * r_s_embed), dim=1)))
            else:
                x_embed = self.bn_entity(self.linear_entity(torch.cat((self.p.alpha * x_g_emb, self.p.beta * x_s_emb), dim=1)))
                r_embed = self.bn_rel(self.linear_rel(torch.cat((self.p.xi * r_g_embed, self.p.mu * r_s_embed), dim=1)))

        else:
            x_embed = x_g_emb
            r_embed = r_g_embed

        sub_emb = x_embed[sub]
        rel_emb = r_embed[rel]

        return sub_emb, rel_emb, x_embed


class SCKGE_TransE(SCKGEBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(SCKGE_TransE, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class SCKGE_DistMult(SCKGEBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class SCKGE_ConvE(SCKGEBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score
