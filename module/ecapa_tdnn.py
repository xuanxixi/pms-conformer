import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import torch.nn as nn
import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
import copy
from spafe.utils import vis
from spafe.features.pncc import pncc
import numpy as np
import librosa
import librosa.display

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        # print("se_x", input.shape)[150, 512, 302])
        x = self.se(input)
        # print("se_x", input.shape)#[150, 512, 302])
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        # width       = 7
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        self.dilation = dilation
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        # print("num_pad",num_pad)
        # num_pad = {}
        # dilation = [2, 3, 4, 5, 6, 2, 3]
        # print("dilation", type(dilation))
        for i in range(self.nums):
            # print("self.nums", self.nums)
            # print("i", i)
            # print("dilation[i]", dilation[i])
            # num_pad[i] = math.floor(kernel_size / 2) * dilation[i]
            # print("num_pad[i]", num_pad[i])
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            # convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation[i], padding=num_pad[i]))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)
        # self.simam     = simam_module(planes)

    def forward(self, x):
        # print("dilation",self.dilation)
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        # print("Bottle2neck-out",out.shape)
        # print("self.width",self.width)
        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
          # print("i",i)
          # print("Bottle2neck-spx[i]", spx[i].shape)
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          #print("Bottle2neck-sp", sp.shape)
          if i==0:
            out = sp
          else:
            # print("out",out.shape)
            # print("sp", sp.shape)
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)
        #print("Bottle2neck-out", out.shape)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        #print("Bottle2neck-out", out.shape)
        out = self.se(out)
        # out = self.simam(out)
        out += residual
        #print("Bottle2neck-out", out.shape)
        return out


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 10), time_mask_width = (0, 5)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class SquaredModulus(nn.Module):
    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = 2 * self._pool(x ** 2.)
        output = output.transpose(1, 2)
        return output

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, n_mels=80, coef=0.97, requires_grad=False):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=False)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

    def forward(self, x):
        # torch.set_printoptions(precision=20)
        # print("x",x.shape)
        x = self.pre_emphasis(x)
        # print("self.window",self.window)
        # print("self.mel_basis",self.mel_basis)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        # x = torch.log(x)
        x = self.instance_norm(x)
        # x = x.unsqueeze(1)
        # print("xend",x.shape)
        return x
class ECAPA_TDNN(nn.Module):

    def __init__(self, C):

        super(ECAPA_TDNN, self).__init__()

        # self.torchfbank = torch.nn.Sequential(
        #     PreEmphasis(),
        #     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        #                                          f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
        #     )
        # # self.torchmel_pcen = torch.nn.Sequential(
        # #     # PreEmphasis(),
        # #     # torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        # #     #                                      f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80),
        # #     PCENTransform(eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True).cuda()
        # #     )
        # self.LEAF = torch.nn.Sequential(
        #     Leaf(n_filters = 80,
        #          sample_rate = 16000,
        #          window_len = 25.,
        #          window_stride = 10.,
        #          preemp = False,
        #          init_min_freq=60.0,
        #          init_max_freq=7800.0,
        #          mean_var_norm = False,
        #          pcen_compression = True,
        #          use_legacy_complex=False,
        #          initializer="default")
        #     )
        # self.torchmel_pcen = torch.nn.Sequential(
        #     # postprocessing.PCENLayer(in_channels=80,
        #     #                          alpha=0.96,
        #     #                          smooth_coef=0.04,
        #     #                          delta=2.0,
        #     #                          root=2.0,  # r=1/root=0.5
        #     #                          floor=1e-12,
        #     #                          trainable=True,
        #     #                          learn_smooth_coef=True,
        #     #                          per_channel_smooth_coef=True)
        #     postprocessing.PCENLayer(in_channels=80,
        #                             alpha=0.98,
        #                             smooth_coef=0.025,
        #                             delta=2.0,
        #                             root=2.0, #r=1/root=0.5
        #                             floor=1e-6,
        #                             trainable=True,
        #                             learn_smooth_coef=True,
        #                             per_channel_smooth_coef=True)
        # )
        # self.cuberoot = torch.nn.Sequential(
        #     # postprocessing.cuberoot(in_channels=80,
        #     #                         alpha=3.00,
        #     #                         trainable=False)
        #     postprocessing.cuberoot(in_channels=80,
        #                             alpha=3.00,
        #                             trainable=True)
        # )
        # # self.torchpncc = torch.nn.Sequential(
        # #     PreEmphasis(),
        # #     torchaudio.transforms.PNCC(sample_rate=16000,num_ceps=80,pre_emph=0,pre_emph_coeff=0.97,power=2,
        # #                                win_len=0.025,win_hop=0.01,win_type="hamming",nfilts=40,#24,
        # #                                nfft=512,low_freq=None,high_freq=None,scale="constant",dct_type=2,
        # #                                use_energy=False,dither=1,lifter=22,normalize=1)
        # #     )
        # # self.torchpncc = torch.nn.Sequential(
        # #     PreEmphasis(),
        # #     pncc(n_fft=512, sr=16000, winlen=0.020, winstep=0.010, n_mels=128, n_pncc=13)
        # # )
        # # self.torchlfcc = torch.nn.Sequential(
        # #     PreEmphasis(),
        # #     torchaudio.transforms.LFCC(sample_rate = 16000,
        # #          n_filter = 128,
        # #          f_min = 0.,
        # #          f_max = None,
        # #          n_lfcc = 40,
        # #          dct_type = 2,
        # #          norm = 'ortho',
        # #          log_lf = False,
        # #          speckwargs = None),
        # # )
        # self.torchMfcc = torch.nn.Sequential(
        #     PreEmphasis(),
        #     torchaudio.transforms.MFCC(sample_rate=16000,
        #                                n_mfcc = 80,
        #                                dct_type = 2,
        #                                norm = 'ortho',
        #                                log_mels = False,
        #                                melkwargs = None),
        # )
        self.mel_trans = Mel_Spectrogram()
        self.specaug = FbankAug() # Spec augmentation

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # self.layer4 = Bottle2neck(C, C, kernel_size=3, dilation=5, scale=8)
        # self.layer5 = nn.Conv1d(4 * C, 1536, kernel_size=1)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            # nn.ReLU(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        # self.bn5 = nn.BatchNorm1d(3072)
        # self.fc6 = nn.Linear(3072, 192)
        # self.bn6 = nn.BatchNorm1d(192)
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 512)
        self.bn6 = nn.BatchNorm1d(512)


    def forward(self, x, aug):#
        # print("x:", x.shape)
        # if model == "train":
        #     # print("1")
        #     x = self.mel_trans(x)
        # elif model == "val":
        #     with torch.no_grad():
        #         # print("2")
        #         x = self.mel_trans(x)
        # with torch.no_grad():
        #     # print("x:", x.shape)
        #     x = self.torchfbank(x)+1e-6
        #     # x = self.torchpncc(x)
        #     #print("fbank:",x.shape)
        #     x = x.log()
        #     # x = self.torchmel_pcen(x)
        #     # print("logfbank:", x.shape)
        #     #x = self.torchlfcc(x) + 1e-6
        #     # x = self.torchMfcc(x) + 1e-6
        #     #print("torchlfcc:", x.shape)
        #     # x = x.log()
        #     #print("logtorchlfcc:", x.shape)
        #     x = x - torch.mean(x, dim=-1, keepdim=True)
        # print("x:", x.shape)
        # x = self.torchmel_pcen(x)
        # print("x:", x.shape)
        # x = self.LEAF(x)
        # x = (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sub(torch.max(x),torch.min(x))
        # x = x - torch.mean(x, dim=-1, keepdim=True)
        # maxx = torch.max(x, dim=-1, keepdim=True)
        # minx = torch.min(x, dim=-1, keepdim=True)
        # x = self.cuberoot(x)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        # sub = torch.sub(maxx, minx)
        # x = x / sub
        # print("x:", x.shape)
        if aug == True:
            x = self.specaug(x)
        # with torch.no_grad():
        #     x = self.torchpncc(x)
        # if aug == True:
        #     x = self.specaug(x)
        # print("fbank:", x.shape)
        # print("pncc:", x.shape)
        # print("x:", x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        #print("x:", x.shape)
        x1 = self.layer1(x)
        # print("x1:", x1.shape)
        x2 = self.layer2(x+x1)
        # print("x2:", x2.shape)
        x3 = self.layer3(x+x1+x2)
        # print("x3:", x3.shape)
        # x4 = self.layer4(x + x1 + x2 + x3)
        # print("x4:", x4.shape)
        # x = self.layer5(torch.cat((x1,x2,x3,x4),dim=1))
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        #print("x:", x.shape)
        #x = torch.sum(x, dim=2)##
        # print("x:", x)
        t = x.size()[-1]
        # print("t:", t)
        # print("torch.mean(x,dim=2,keepdim=True).repeat(1,1,t)",torch.mean(x,dim=2,keepdim=True).repeat(1,1,t).shape)
        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        # print("global_x:", global_x.shape)
        w = self.attention(global_x)
        # print("w:", w.shape)
        # print("x * w",(x * w).shape)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        # print("mu:", mu.shape)
        # print("sg:", sg.shape)
        x = torch.cat((mu,sg),1)
        # print("x:", x.shape)
        x = self.bn5(x)##
        x = self.fc6(x)##
        x = self.bn6(x)##
        #print("x:", x.shape)
        return x

# class Model(torch.nn.Module):
#     def __init__(self, n_mels=80, embedding_dim=192, channel=512):
#         super(Model, self).__init__()
#         channels = [channel for _ in range(4)]
#         channels.append(channel*3)
#         self.specaug = FbankAug()
#         self.model = ECAPA_TDNN(input_size=n_mels, lin_neurons=embedding_dim, channels=channels)
#
#     def forward(self, x, aug):
#         feat = x.squeeze(1)
#         feat = feat - torch.mean(feat, dim=-1, keepdim=True)
#         if aug == True:
#             x = self.specaug(feat)
#         x = x.permute(0, 2, 1)
#         x = self.model(x)
#         x = x.squeeze(1)
#         return x
 
def ecapa_tdnn(n_mels=80, embedding_dim=192, channel=512):
    model = ECAPA_TDNN(C = channel)
    return model

def ecapa_tdnn_large(n_mels=80, embedding_dim=512, channel=1024):
    model = ECAPA_TDNN(C = channel)
    return model

# for i in range(6, 7):
#     print("num_blocks is {}".format(i))
#     model = ecapa_tdnn_large(n_mels=80, embedding_dim=512, channel=1024)
#
#     import time
#     model = model.eval()
#     time1 = time.time()
#     with torch.no_grad():
#         for i in range(100):
#             data = torch.randn(1,  80, 300)
#             embedding = model(data)
#     time2 = time.time()
#     val = (time2 - time1)/100
#     rtf = val / 5
#
#     total = sum([param.nelement() for param in model.parameters()])
#     print("total param: {:.2f}M".format(total/1e6))
#     print("RTF {:.4f}".format(rtf))
