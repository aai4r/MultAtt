import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer

class Face_encoding(nn.Module):
    def __init__(self):
        super(Face_encoding, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class Context_encoding(nn.Module):
    def __init__(self):
        super(Context_encoding, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.A1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.A2 = nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))

        x1 = F.relu(self.bn6(self.A1(x)))
        x1 = F.relu(self.bn7(self.A2(x1)))
        x1 = F.softmax(x1.view(x1.shape[0], -1), dim=1)
        x1 = x1.reshape(x1.shape[0], 1, 14, 14)

        x = x * x1
        return x

class AdaptiveFusion(nn.Module):
    def __init__(self):
        super(AdaptiveFusion, self).__init__()
        self.Avg_F = nn.AvgPool2d(2)
        self.Avg_C = nn.AvgPool2d(2)

        self.A0_F = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.A1_F = nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.A0_C = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.A1_C = nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn1_F = nn.BatchNorm2d(128)
        self.bn2_F = nn.BatchNorm2d(1)
        self.bn1_C = nn.BatchNorm2d(128)
        self.bn2_C = nn.BatchNorm2d(1)

        self.L0 = nn.Linear(512 * 7 * 7, 128)
        self.Lfinal = nn.Linear(128, 7)


    def forward(self, x1, x2):
        x1 = self.Avg_F(x1)
        x2 = self.Avg_C(x2)

        A1 = F.relu(self.bn1_F(self.A0_F(x1)))
        A1 = F.relu(self.bn2_F(self.A1_F(A1)))

        A2 = F.relu(self.bn1_C(self.A0_C(x2)))
        A2 = F.relu(self.bn2_C(self.A1_C(A2)))

        A = F.softmax(torch.cat((A1.view(A1.shape[0], -1), A2.view(A2.shape[0], -1)), dim=1), dim=1)
        A1 = A[:, :49].reshape(A1.shape[0], 1, 7, 7)
        A2 = A[:, 49:].reshape(A2.shape[0], 1, 7, 7)

        x1 = x1 * A1
        x2 = x2 * A2
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.L0(x))
        x = self.Lfinal(x)

        return x

class MultiTransformer(nn.Module):
    def __init__(self):
        super(MultiTransformer, self).__init__()
        self.Transformer1 = Transformer(256, 64, 4)

        self.Avg_F = nn.AvgPool2d(2)
        self.Avg_C = nn.AvgPool2d(2)

        self.L0 = nn.Linear(512 * 7 * 7, 4096)
        self.Lfinal = nn.Linear(4096, 7)

    def forward(self, x1, x2):
        x1, x2 = self.Transformer1(x1, x2)

        x1 = self.Avg_F(x1)
        x2 = self.Avg_C(x2)

        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.shape[0], -1)

        x = F.relu(self.L0(x))
        x = self.Lfinal(x)

        return x


class CAERnet(nn.Module):
    def __init__(self):
        super(CAERnet, self).__init__()
        self.face_features = Face_encoding()
        self.context_features = Context_encoding()
        self.adaptive_fusion = AdaptiveFusion()

    def forward(self, x1, x2):
        x1 = self.face_features(x1)
        x2 = self.context_features(x2)

        x = self.adaptive_fusion(x1, x2)
        return x


class OURnet(nn.Module):
    def __init__(self):
        super(OURnet, self).__init__()
        self.face_features = Face_encoding()
        self.context_features = Context_encoding()
        self.multi_transformer = MultiTransformer()

    def forward(self, x1, x2):
        x1 = self.face_features(x1)
        x2 = self.context_features(x2)

        x = self.multi_transformer(x1, x2)
        return x

