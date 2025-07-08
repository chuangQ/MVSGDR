import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, v):
        q_attn = torch.matmul(q / self.temperature, v.transpose(1, 2))
        k_attn = torch.matmul(v / self.temperature, k.transpose(1, 2))
        q_attn = q_attn.mean(dim=1, keepdim=True)
        k_attn = k_attn.mean(dim=1, keepdim=True)
        q_attn = F.softmax(q_attn, dim=-1)
        k_attn = F.softmax(k_attn, dim=-1)

        return q_attn, k_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, channels):
        super(MultiHeadAttention, self).__init__()

        self.channels = channels

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=channels ** 0.5)

    def forward(self, q, k, v, layer_num):
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        q_attn, k_attn = self.attention(q, k, v)

        # hop-attention
        attn_score1, topk_indices1 = torch.topk(q_attn[0], k=3)
        attn_mask1 = torch.zeros(layer_num).to('cuda')
        attn_mask1[topk_indices1[0]] = 1
        result1 = torch.zeros_like(q_attn[0][0])
        result1.scatter_(0, topk_indices1[0], attn_score1[0])
        result1 = result1 * attn_mask1

        attn_score2, topk_indices2 = torch.topk(k_attn[0], k=3)
        attn_mask2 = torch.zeros(layer_num).to('cuda')
        attn_mask2[topk_indices2[0]] = 1
        result2 = torch.zeros_like(k_attn[0][0])
        result2.scatter_(0, topk_indices2[0], attn_score2[0])
        result2 = result2 * attn_mask2

        return result1, result2


class HopLayer_atten(nn.Module):
    def __init__(self, channels):
        super(HopLayer_atten, self).__init__()
        self.layer_transformer = MultiHeadAttention(channels)
        self.attn = None
        self.fc = nn.Linear(channels, channels, bias=False)

    def forward(self, all_layer):
        combined_embedding = torch.stack(all_layer, dim=0)
        layer_embedding = combined_embedding.mean(dim=1, keepdim=False).view(len(all_layer), -1)
        p = layer_embedding.unsqueeze(0)
        score1, score2 = self.layer_transformer(p, p, p, len(all_layer))

        adap_layer_u = torch.einsum('i,ibc->bc', score1, combined_embedding)
        adap_layer_v = torch.einsum('i,ibc->bc', score2, combined_embedding)

        return self.fc(adap_layer_u + adap_layer_v)
