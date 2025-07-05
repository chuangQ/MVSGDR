from layer.BIST import *
from layer.Force_GNN import GNN_Fnormal
from layer.MVS import MVSGCN
from layer.NSB import NSB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MVSGDR(nn.Module):
    def __init__(self, args):
        super(MVSGDR, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.drug_num = args.drug_num
        self.disease_num = args.disease_num
        self.total_num = args.total_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.co_embedding_dim = args.co_embedding_dim
        self.hidden_embedding_dim = args.hidden_embedding_dim

        self.fuse_p = nn.Linear(self.co_embedding_dim, self.co_embedding_dim)

        self.force_dr = GNN_Fnormal(self.drug_num, self.hidden_embedding_dim, self.co_embedding_dim)
        self.force_di = GNN_Fnormal(self.disease_num, self.hidden_embedding_dim, self.co_embedding_dim)
        self.force_co = GNN_Fnormal(self.co_embedding_dim, self.hidden_embedding_dim, self.co_embedding_dim,
                                    local_layers=5, dropout=0.5)

        self.nsb = NSB(self.co_embedding_dim)
        self.bilevel_net = BiLevel_net(self.total_num, in_channels=args.co_embedding_dim,
                                       hidden_channels=args.co_embedding_dim,
                                       out_channels=args.co_embedding_dim, layers=2, n_head=8)

        self.mvssc_net = MVSGCN(in_channels=self.co_embedding_dim, out_channels=self.co_embedding_dim, gcn_nums=5,
                                g_name="GCN", sz_c=4, graph_norm=True, gcn_h_dim=self.co_embedding_dim, device='cuda',
                                mask_rate=0.4, MFeatype='MFM')

        self.mlp = nn.Sequential(
            nn.Linear(args.co_embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 2)
        )

    def forward(self, drdr_g, didi_g, dr_sim, di_sim, pos_sample, patch2, neg_sample):
        # coarse feature
        drdr_edge = torch.stack((drdr_g.edges()[0], drdr_g.edges()[1]), dim=0)
        didi_edge = torch.stack((didi_g.edges()[0], didi_g.edges()[1]), dim=0)
        dr_coarse_f = self.force_dr(dr_sim, drdr_edge)
        di_coarse_f = self.force_di(di_sim, didi_edge)

        # dr-di graph
        coarse_f = torch.cat((dr_coarse_f, di_coarse_f), dim=0)
        edge_index = pos_sample.transpose(0, 1).clone()
        edge_index[1] += self.drug_num

        # metis and BIST
        z1 = self.bilevel_net(coarse_f, patch2, need_attn=False)
        z_r1 = z1[:self.drug_num, :]
        z_d1 = z1[self.drug_num:]
        bi_f = torch.cat((z_r1, z_d1), dim=0)
        bi_hat_f = self.force_co(bi_f, edge_index)
        dr_f = bi_hat_f[:self.drug_num, :]
        di_f = bi_hat_f[self.drug_num:]
        bist_dr = 0.5 * z_r1 + 0.5 * dr_f
        bist_di = 0.5 * z_d1 + 0.5 * di_f

        ###################################
        # cross_feat
        cross_sub = torch.cat((z_r1, z_d1))
        cross_gcn = torch.cat((bist_dr, bist_di))
        ###################################

        # MVS
        batch = torch.full((self.total_num,), 0, dtype=torch.int64).to(device)
        mv = self.mvssc_net(coarse_f, edge_index, batch, cross_sub, cross_gcn)
        dr_mv = mv[:self.drug_num, :]
        di_mv = mv[self.drug_num:]

        # net_combine
        combine_r = self.alpha * dr_mv + (1 - self.alpha) * bist_dr
        combine_d = self.alpha * di_mv + (1 - self.alpha) * bist_di

        # predict
        pos_link = torch.mul(combine_r[pos_sample[:, 0]], combine_d[pos_sample[:, 1]])
        neg_link = torch.mul(combine_r[neg_sample[:, 0]], combine_d[neg_sample[:, 1]])
        neg_link_b, _ = self.neg_fusion(pos_sample, neg_sample, combine_r, combine_d)

        drdi_emb = torch.cat((pos_link, neg_link_b), dim=0)
        output_all = self.mlp(drdi_emb)

        return output_all, pos_link, neg_link, neg_link_b

    def loss_neg_balance(self, pred, neg_pred, neg_pred_b, label):
        # predict loss
        pre_loss = F.cross_entropy(pred, label)
        # KL calculate
        kl_divergence1 = F.kl_div(F.log_softmax(neg_pred, dim=1), F.softmax(neg_pred_b, dim=1), reduction='batchmean')
        kl_divergence2 = F.kl_div(F.log_softmax(neg_pred_b, dim=1), F.softmax(neg_pred, dim=1), reduction='batchmean')
        neg_loss = 0.5 * (kl_divergence1 + kl_divergence2)
        loss = pre_loss + self.beta * neg_loss
        return loss

    def neg_fusion(self, pos_sample, neg_sample, emb_r, emb_d):
        # pos emb
        emb_p_r = emb_r[pos_sample[:, 0]]
        emb_p_d = emb_d[pos_sample[:, 1]]
        # neg emb
        emb_n_r = emb_r[neg_sample[:, 0]]
        emb_n_d = emb_d[neg_sample[:, 1]]
        # calcu pos mean and neg mean
        emb_p_link = torch.mul(emb_p_r, emb_p_d)
        p_mean = emb_p_link.mean(dim=0)
        emb_n_link = torch.mul(emb_n_r, emb_n_d)
        n_mean = emb_n_link.mean(dim=0)
        # cosine sim
        sim_p = F.cosine_similarity(emb_n_link, p_mean, dim=1)
        sim_n = F.cosine_similarity(emb_n_link, n_mean, dim=1)

        closer_to_p_mask = sim_p > sim_n
        closer_to_p_mask_pn = sim_p > 0
        cosin_p_mask = closer_to_p_mask.unsqueeze(1)
        cosin_p_mask_pn = closer_to_p_mask_pn.unsqueeze(1)

        # pearson sim
        p_corr = self.pearson_correlation(emb_n_link, p_mean)
        n_corr = self.pearson_correlation(emb_n_link, n_mean)
        pi_mask = (p_corr > 0).unsqueeze(1).to('cuda')
        pi_mask_pn = (p_corr > n_corr).unsqueeze(1).to('cuda')

        # combine sim
        pi_co_mask = torch.logical_and(pi_mask, pi_mask_pn)
        cosin_co_mask = torch.logical_and(cosin_p_mask, cosin_p_mask_pn)
        co_mask = torch.logical_and(cosin_co_mask, pi_co_mask)

        # condition
        p_mean_a = self.nsb(p_mean)
        fused_embeddings = emb_n_link + co_mask * self.fuse_p(p_mean_a)

        return fused_embeddings, co_mask

    def pearson_correlation(self, x, y):
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean()

        cov = (x - x_mean) * (y - y_mean.view(1, -1))

        x_std = (x - x_mean).pow(2).mean(dim=1, keepdim=True).sqrt()
        y_std = (y - y_mean.view(1, -1)).pow(2).mean(dim=1, keepdim=True).sqrt()

        corr = cov.mean(dim=1, keepdim=True) / (x_std * y_std)
        return corr.squeeze()
