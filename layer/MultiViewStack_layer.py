import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from layer.DataTransform import feature_mask
import numpy as np
from layer.Hop_attention import HopLayer_atten
from layer.Force_GNN import GNN_Fnormal
import torch.nn.functional as F

gnn_dict = {
    "GCN": GNN_Fnormal,
    "GraphSAGE": GNN_Fnormal
}


class MultiViewStackLayers(nn.Module):

    def __init__(self, d_in, d_h, d_out, sz_c, sz_l, mask_rate, device, g_name="GCN",
                 g_norm=False, is_em=False, MFeatype='SFM'):
        super().__init__()
        self.MFeatype = MFeatype
        self.mask_rate = mask_rate
        self.is_em = is_em
        self.sz_c = sz_c
        d_in = d_in
        d_h = d_h
        self.d_out = d_out
        self.sz_l = sz_l

        self.device = device

        self.hop_atten = HopLayer_atten(d_out)

        if self.MFeatype == 'MFM':
            rates = [rate / 10 for rate in range(int(mask_rate * 10 + 1))]
            self.all_mask_rate = np.random.choice(rates, self.sz_c, replace=True)
        g_name = g_name
        self.gcn_layer = nn.ModuleList(self.channal_block(d_in, d_h, d_out, g_name, g_norm))
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.gnn_view0 = GNN_Fnormal(d_out, d_out, d_out, ln=True, bn=True, local_layers=5, dropout=0.5)
        self.gnn_view1 = GNN_Fnormal(d_out, d_out, d_out, ln=True, bn=True, local_layers=5, dropout=0.5)
        self.gnn_view2 = GNN_Fnormal(d_out, d_out, d_out, ln=True, bn=True, local_layers=5, dropout=0.5)
        self.gnn_view3 = GNN_Fnormal(d_out, d_out, d_out, ln=True, bn=True, local_layers=5, dropout=0.5)

        self.cross0 = nn.Linear(d_in, d_out)
        self.cross1 = nn.Linear(d_in, d_out)

    def reset_parameters(self):
        for layer in self.gcn_layer:
            for f in layer:
                if type(f) in list(gnn_dict.values()):
                    f.reset_parameters()
                if type(f) == gnn.BatchNorm:
                    f.reset_parameters()

    def channal_block(self, d_in, d_h, d_out, g_name, g_norm):
        layer = []
        # 4 channel
        for c in range(self.sz_c):
            t_layer = []
            # n gnn layer in per channel
            for l in range(self.sz_l):
                if l == 0:
                    t_layer.append(self.get_gnn(d_in, d_h, g_name))
                elif l == (self.sz_l - 1):
                    t_layer.append(self.get_gnn(d_h, d_out, g_name))
                else:
                    t_layer.append(self.get_gnn(d_h, d_h, g_name))
                t_layer.append(nn.ReLU())
                t_layer.append(nn.Dropout(0.0))
                if g_norm:
                    t_layer.append(gnn.GraphSizeNorm())
                    if l == (self.sz_l - 1):
                        t_layer.append(gnn.BatchNorm(d_out))
                    else:
                        t_layer.append(gnn.BatchNorm(d_h))
                t_layer.append(nn.ReLU())
            layer.append(nn.ModuleList(t_layer))
        return layer

    def get_gnn(self, d_in, d_out, g_name):
        return gnn_dict[g_name](d_in, d_out * 2, d_out, gnn='gcn', dropout=0.3, local_layers=2)

    def fea_mask(self, x, i):
        if self.MFeatype == 'SFM':
            h = feature_mask(x, self.mask_rate, self.device)
        elif self.MFeatype == 'TFM':
            if self.training:
                h = feature_mask(x, self.mask_rate, self.device)
            else:
                h = x
        elif self.MFeatype == "MFM":
            if self.training:
                h = feature_mask(x, self.all_mask_rate[i], self.device)
            else:
                h = x
        else:
            raise NotImplementedError
        return h

    def forward(self, x, edge, batch, cross_feature0, cross_feature1):
        drdi_view = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
        # create channel e.p 4
        for i, layer in enumerate(self.gcn_layer):
            h = self.fea_mask(x, i)
            all_layer = []
            for f in layer:
                if type(f) in list(gnn_dict.values()):
                    h = f(h, edge)
                    all_layer.append(h)
                elif type(f) == gnn.GraphSizeNorm:
                    h = f(h, batch)
                else:
                    h = f(h)

            # hop level attention
            h_f = all_layer[0]
            hop_layer = all_layer[1:]
            # hop_layer = all_layer
            adap_layer = self.hop_atten(hop_layer)
            h_f = h_f + adap_layer
            drdi_view[i] = h_f

        cross_feature0 = self.cross0(cross_feature0)
        cross_feature1 = self.cross1(cross_feature1)

        # multi view stack
        drdi_view[1] = self.gnn_view1(drdi_view[0] + drdi_view[1], edge)
        drdi_view[2] = self.gnn_view2(drdi_view[1] + drdi_view[2] + cross_feature0, edge)
        drdi_view[3] = self.gnn_view3(drdi_view[2] + drdi_view[3] + cross_feature1, edge)

        return drdi_view
