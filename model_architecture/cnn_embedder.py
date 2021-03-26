import torch
from torch import nn

class BNLayer(nn.Module):
    def __init__(self, f_in, f_out, k, s, p=0, r=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(f_in, f_out, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(f_out)
        self.r = r
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.ReLU()(x) if self.r else x
        return x
class Layer(nn.Module):
    def __init__(self, f_in, f_out, s=2, r=True):
        super().__init__()
        self.bnl1 = BNLayer(f_in, f_out, 3, s, 1)
        self.bnl2 = BNLayer(f_out, f_out, 3, 1, 1, r=False)
        self.bnl3 = BNLayer(f_out, f_out, 3, 1, 1)
        self.bnl4 = BNLayer(f_out, f_out, 3, 1, 1, r=False)
        self.s = s
        self.r = r
        self.pool = BNLayer(f_in,f_in,3,1,1) if s==1 else nn.MaxPool2d(2,s,0)
    def forward(self, x_in):
        x = self.bnl1(x_in)
        x = self.bnl2(x) + self.bnl4(self.bnl3(x))
        x_pool = self.pool(x_in)
        x = torch.cat((x, x_pool), dim=1) if self.r else x
        return x
class CNNEmbedder(nn.Module):
    def __init__(self, d_model, N, n):
        super().__init__()
        self.bnl1 = BNLayer(1, N, 3, 2, 1)
        self.layer1 = Layer(N        , n)
        self.layer2 = Layer(n + N    , n)
        self.layer3 = Layer(2 * n + N, n)
        self.layer4 = Layer(3 * n + N, n, s=2)
        self.layer5 = Layer(4 * n + N, d_model, s=1, r=False)
    def forward(self, x):
        x = self.bnl1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x