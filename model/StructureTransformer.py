import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.open3dvis import get_colors_with_tsne
import numpy as np


class Attention(nn.Module):
    def __init__(self, d_model, h):
        super(Attention, self).__init__()
        self.d_model_pre_head = d_model // h

    def forward(self, Q, K, V, R):
        QK = torch.matmul(Q, K.transpose(1, 2))
        QR = torch.matmul(Q, R.transpose(1, 2))
        attention_score = (QK + QR) / self.d_model_pre_head ** 0.5

        return torch.matmul(torch.softmax(attention_score, dim=2), V),attention_score


class MultiHead(nn.Module):
    def __init__(self, d_model=256, h=4):
        super(MultiHead, self).__init__()

        self.h = h
        self.WQ = nn.ModuleList([nn.Linear(d_model, d_model // h) for _ in range(h)])
        self.WK = nn.ModuleList([nn.Linear(d_model, d_model // h) for _ in range(h)])
        self.WV = nn.ModuleList([nn.Linear(d_model, d_model // h) for _ in range(h)])
        self.WR = nn.ModuleList([nn.Linear(d_model, d_model // h) for _ in range(h)])
        self.WO = nn.Linear(h * (d_model // h), d_model)
        self.head = nn.ModuleList([Attention(d_model, h) for _ in range(h)])

    def forward(self, Q, K, V, R):
        attention_score = {}
        x=[]
        for i in range(self.h):
            output,score=self.head[i](self.WQ[i](Q), self.WK[i](K), self.WV[i](V), self.WR[i](R))
            x.append(output)
            attention_score['{}'.format(i)]=score
        x=torch.cat(x,dim=-1)
        return self.WO(x) ,attention_score # (B,N,256)


class Transformer(nn.Module):
    def __init__(self, d_model=256, h=4):
        super(Transformer, self).__init__()
        self.attention = MultiHead(d_model, h)
        self.ffn1 = nn.Linear(d_model, d_model * 2)
        self.ffn2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, Fx, Fy):
        """
        :param Fx: torch.Tensor (B, N, C)
        :param Fy: torch.Tensor (B, N, C)
        :return:  output: torch.Tensor (B, N, C)
        """
        attention_output,attention_score = self.attention(Fx, Fx, Fx, Fy)
        output1 = self.layernorm1(attention_output + Fx)
        output2 = F.relu(self.ffn1(output1))
        output2 = self.ffn2(output2)
        output2 = self.dropout(output2)
        output = self.layernorm2(output1 + output2)
        output = output/torch.norm(output, dim=-1, keepdim=True)
        return output,attention_score

