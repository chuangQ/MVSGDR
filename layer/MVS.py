import torch.nn.functional as F
from torch import nn
import torch
from layer.MultiViewStack_layer import MultiViewStackLayers
import utils


class MVSGCN(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, gcn_nums, g_name, sz_c, graph_norm, gcn_h_dim, device,
                 mask_rate=0.0, is_em=False, MFeatype='SFM'):
        super(MVSGCN, self).__init__()
        # *******************************def params******************************
        self.device = device
        gcn_h_dim = gcn_h_dim // sz_c
        self.sz_c = sz_c

        # *******************************def models******************************
        self.is_em = is_em
        if is_em:
            self.fea_embed = nn.Sequential(nn.Linear(in_channels, gcn_h_dim))
            in_channels = gcn_h_dim
        self.mvslayer = MultiViewStackLayers(in_channels, gcn_h_dim, gcn_h_dim, sz_c, gcn_nums, mask_rate=mask_rate,
                                             device=self.device, g_name=g_name, g_norm=graph_norm, MFeatype=MFeatype)

    def reset_parameters(self):
        if self.is_em:
            self.fea_embed.apply(utils.weight_reset)
        if self.smus is not None:
            for m in self.smus:
                m.reset_parameters()
        all_res = [self.mvslayer]
        for res in all_res:
            if res != None:
                res.reset_parameters()

    def forward(self, data, edge_index, batch, con_bin, con_gcn):
        if self.is_em:
            data = self.fea_embed(data)
        # multi-channel encoder
        z = self.mvslayer(data, edge_index, batch, con_bin, con_gcn)
        # multi-channel n->1
        z_feat2 = z.transpose(0, 1).reshape(data.size(0), data.size(1))

        return z_feat2
