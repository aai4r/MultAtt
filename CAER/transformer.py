import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv1x1(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)

def conv3x3(inplanes, outplanes):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)

class SHattention(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SHattention, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes

        self.makeQ1 = conv1x1(self.inplanes, self.outplanes)
        self.makeK1 = conv1x1(self.inplanes, self.outplanes)
        self.makeV1 = conv1x1(self.inplanes, self.outplanes)

        self.makeQ2 = conv1x1(self.inplanes, self.outplanes)
        self.makeK2 = conv1x1(self.inplanes, self.outplanes)
        self.makeV2 = conv1x1(self.inplanes, self.outplanes)

    def forward(self, x1, x2):
        Q1 = self.makeQ1(x1)
        K1 = self.makeK1(x1)
        V1 = self.makeV1(x1)

        Q2 = self.makeQ2(x1)
        K2 = self.makeK2(x1)
        V2 = self.makeV2(x1)

        score1_1 = (Q1 * K1).sum(dim=1) / math.sqrt(self.outplanes)
        score1_2 = (Q1 * K2).sum(dim=1) / math.sqrt(self.outplanes)

        score1 = F.softmax(torch.cat((score1_1.view(Q1.shape[0], -1), score1_2.view(Q1.shape[0], -1)), dim=1), dim=1)
        score1_1 = score1[:, :196].reshape(score1_1.shape[0], 1, 14, 14)
        score1_2 = score1[:, 196:].reshape(score1_2.shape[0], 1, 14, 14)

        Z1 = score1_1 * V1 + score1_2 * V2

        score2_1 = (Q2 * K1).sum(dim=1) / math.sqrt(self.outplanes)
        score2_2 = (Q2 * K2).sum(dim=1) / math.sqrt(self.outplanes)

        score2 = F.softmax(torch.cat((score2_1.view(Q2.shape[0], -1), score2_2.view(Q2.shape[0], -1)), dim=1), dim=1)
        score2_1 = score2[:, :196].reshape(score2_1.shape[0], 1, 14, 14)
        score2_2 = score2[:, 196:].reshape(score2_2.shape[0], 1, 14, 14)

        Z2 = score2_1 * V1 + score2_2 * V2

        return Z1, Z2


class MHattention(nn.Module):
    def __init__(self, inplanes, outplanes, multi_head):
        super(MHattention, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.multi_head = multi_head
        for MH in range(self.multi_head):
            SH_name = 'SH' + str(MH + 1)
            setattr(self, SH_name, SHattention(inplanes, outplanes))

        self.makeZ1 = conv1x1(self.outplanes * self.multi_head, self.inplanes)
        self.makeZ2 = conv1x1(self.outplanes * self.multi_head, self.inplanes)

    def forward(self, x1, x2):
        Z1, Z2 = self.SH1(x1, x2)
        for MH in range(self.multi_head - 1):
            SH_name = 'SH' + str(MH + 2)
            z1, z2 = getattr(self, SH_name)(x1, x2)
            Z1, Z2 = torch.cat((Z1, z1), dim=1), torch.cat((Z2, z2), dim=1)

        Z1 = self.makeZ1(Z1)
        Z2 = self.makeZ1(Z2)

        return Z1, Z2

class Transformer(nn.Module):
    def __init__(self, inplanes, outplanes, multi_head):
        super(Transformer, self).__init__()
        self.MH = MHattention(inplanes, outplanes, multi_head)
        self.FF1 = conv3x3(inplanes, inplanes)
        self.FF2 = conv3x3(inplanes, inplanes)

        self.LN1_1 = nn.LayerNorm([inplanes, 14, 14])
        self.LN1_2 = nn.LayerNorm([inplanes, 14, 14])

        self.LN2_1 = nn.LayerNorm([inplanes, 14, 14])
        self.LN2_2 = nn.LayerNorm([inplanes, 14, 14])

    def forward(self, x1, x2):
        z1, z2 = self.LN1_1(x1), self.LN1_2(x2)
        z1, z2 = self.MH(z1, z2)

        z1 += x1
        z2 += x2

        Z1, Z2 = self.LN2_1(z1), self.LN2_2(z2)
        Z1, Z2 = self.FF1(Z1), self.FF2(Z2)

        Z1 += z1
        Z2 += z2

        return Z1, Z2



