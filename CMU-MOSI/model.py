from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Subnet(nn.Module):
    def __init__(self, input_dim):
        super(Subnet, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.post_fusion_layer_1 = nn.Linear(input_dim, 128)
        self.post_fusion_layer_2 = nn.Linear(128, 32)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.post_fusion_layer_1(x))
        x = F.relu(self.post_fusion_layer_2(x))

        return x

class LSTMSubNet(nn.Module):
    def __init__(self, in_size, hidden_size_1, out_size, dropout):
        super(LSTMSubNet, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.lstm1 = nn.LSTMCell(in_size, hidden_size_1)
        # self.lstm2 = nn.LSTMCell(hidden_size, hidden_size_1)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size_1, out_size)

    def forward(self, x):
        n = x.shape[1]
        t = x.shape[0]

        first_h = torch.zeros(n, self.hidden_size_1).to(device)
        first_c = torch.zeros(n, self.hidden_size_1).to(device)

        all_hs_1 = []
        h, c = first_h, first_c
        for i in range(t):
            new_h, new_c = self.lstm1(x[i], (h, c))
            h, c = new_h, new_c
            all_hs_1.append(h)

        # all_hs_2 = []
        # h, c = first_h, first_c
        # for i in range(t):
        #     new_h, new_c = self.lstm2(all_hs_1[i], (h, c))
        #     h, c = new_h, new_c
        #     all_hs_2.append(h)

        x = all_hs_1[-1]
        # x = all_hs_2[-1]
        x = self.dropout(x)
        x = F.relu(self.linear_1(x))

        return x

class SHattention(nn.Module):
    def __init__(self, hidden_dim):
        super(SHattention, self).__init__()
        self.hidden_dim = hidden_dim
        self.makeQ = nn.Linear(32, hidden_dim)
        self.makeK = nn.Linear(32, hidden_dim)
        self.makeV = nn.Linear(32, hidden_dim)

    def forward(self, x):
        Q = self.makeQ(x)
        K = self.makeK(x)
        V = self.makeV(x)

        Z = torch.zeros_like(V)
        for n in range(int(Q.shape[0])):
            score = F.softmax(torch.matmul(Q[n], torch.transpose(K[n], 0, 1)), dim=1) / math.sqrt(self.hidden_dim)
            z = torch.matmul(score, V[n])
            Z[n] = z
        return Z

class MHattention(nn.Module):
    def __init__(self, hidden_dim, multi_head):
        super(MHattention, self).__init__()
        self.hidden_dim = hidden_dim
        self.multi_head = multi_head
        for MH in range(self.multi_head):
            SH_name = 'SH' + str(MH + 1)
            setattr(self, SH_name, SHattention(hidden_dim))

        self.makeZ = nn.Linear(self.hidden_dim * self.multi_head, 32)

    def forward(self, x):
        Z = self.SH1(x)
        for MH in range(self.multi_head - 1):
            SH_name = 'SH' + str(MH + 2)
            z = getattr(self, SH_name)(x)
            Z = torch.cat((Z, z), dim=2)

        Z = self.makeZ(Z)
        return Z

class Transformer(nn.Module):
    def __init__(self, hidden_dim, multi_head):
        super(Transformer, self).__init__()
        self.MH = MHattention(hidden_dim, multi_head)
        self.FF = nn.Linear(32, 32)

        self.LN1 = nn.LayerNorm([3, 32])
        self.LN2 = nn.LayerNorm([3, 32])

    def forward(self, x):
        z = self.LN1(x)
        z = self.MH(z)

        z += x

        Z = self.LN2(z)
        Z = self.FF(Z)

        Z += z

        return Z

class PF(nn.Module):
    def __init__(self, input_dim):
        super(PF, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.post_fusion_layer_1 = nn.Linear(input_dim, 32)
        self.post_fusion_layer_2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.post_fusion_layer_1(x))
        x = F.softmax(self.post_fusion_layer_2(x), dim=1)
        return x

class Ours(nn.Module):
    def __init__(self):
        super(Ours, self).__init__()
        self.LSubnet = Subnet(768)
        self.ASubnet = LSTMSubNet(74, 32, 32, 0.1)
        self.VSubnet = LSTMSubNet(47, 32, 32, 0.1)

        self.Transformer1 = Transformer(8, 4)

        self.PostFusion = PF(32 * 3)

    def forward(self, l, a, v):
        l = self.LSubnet(l)
        a = self.ASubnet(a)
        v = self.VSubnet(v)

        x = torch.cat((l.unsqueeze(1), a.unsqueeze(1), v.unsqueeze(1)), dim=1)
        x = self.Transformer1(x)

        x = x.reshape(x.shape[0], -1)
        x = self.PostFusion(x)

        return x

