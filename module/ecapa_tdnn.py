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

 
def ecapa_tdnn(n_mels=80, embedding_dim=192, channel=512):
    model = ECAPA_TDNN(C = channel)
    return model

def ecapa_tdnn_large(n_mels=80, embedding_dim=512, channel=1024):
    model = ECAPA_TDNN(C = channel)
    return model


