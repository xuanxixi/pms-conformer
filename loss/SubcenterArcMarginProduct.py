
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from .utils import accuracy

class SubcenterArcMarginProduct(nn.Module):
    r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
        """

    def __init__(self, in_features, out_features, K=2, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ce = nn.CrossEntropyLoss()
        self.s = s
        self.m = m
        self.K = K
        self.weight = Parameter(torch.FloatTensor(out_features * self.K, in_features))
        nn.init.xavier_uniform_(self.weight)

        print('Initialised SC-AAM-Softmax m=%.3f s=%.3f' % (self.m, self.s))
        print('Embedding dim is {}, number of speakers is {}'.format( in_features, out_features))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # print("F.normalize(input)",F.normalize(input).shape)#torch.Size([150, 192])
        # print("F.normalize(weight)", F.normalize(self.weight).shape)#torch.Size([2422, 192])
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.K > 1:
            # print("cosine1",cosine.shape)#([150, 2422])
            cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
            # print("cosine2", cosine.shape)#([150, 1211, 2])
            cosine, _ = torch.max(cosine, axis=2)
            # print("torch.max(cosine, axis=2)",torch.max(cosine, axis=2))
            # print("cosine3", cosine.shape)#([150, 1211])

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(phi+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # print("1")
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # print("2")

            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
        # return output