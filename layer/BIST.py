import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.to('cuda:0')
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)

        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.dropout(F.relu(self.fc(q)))
        q = q + residual

        return q, attn


class FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(F.relu(self.lin1(x)))
        x = self.Dropout(self.lin2(x))
        x = x + residual
        return x


class BiLayer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(BiLayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)
        self.patch_ffn = FFN(channels, dropout)
        self.fuse_lin = nn.Linear(2 * channels, channels)
        self.use_patch_attn = use_patch_attn
        self.attn = None

    def forward(self, x, patch, attn_mask=None, need_attn=False):

        x = self.node_norm(x)

        patch_x = x[patch]
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)
        if need_attn:
            self.attn = torch.zeros((x.shape[0], x.shape[0]))
            for i in tqdm(range(patch.shape[0])):
                p = patch[i].tolist()
                row = torch.tensor([p] * len(p)).T.flatten()
                col = torch.tensor(p * len(p))
                a = attn[i].mean(0).flatten().cpu()
                self.attn = self.attn.index_put((row, col), a)

            self.attn = self.attn[:-1][:, :-1].detach().cpu()

        if self.use_patch_attn:
            p = self.patch_norm(patch_x.mean(dim=1, keepdim=False)).unsqueeze(0)
            p, _ = self.patch_transformer(p, p, p)
            p = self.patch_ffn(p).permute(1, 0, 2)
            #
            p = p.repeat(1, patch.shape[1], 1)
            z = torch.cat([patch_x, p], dim=2)
            patch_x = F.relu(self.fuse_lin(z)) + patch_x
        x[patch] = patch_x

        return x


class BiLevel_net(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BiLevel_net, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels)
        self.BiLayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BiLayers.append(
                BiLayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn = []

    def forward(self, x: torch.Tensor, patch: torch.Tensor, need_attn=False):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BiLayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BiLayers[i].attn)
        x = self.classifier(x)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
