import torch
from wenet.transformer.encoder_cat import Squeeze_ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d
import torch.nn as nn
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
class Conformer(torch.nn.Module):
    def __init__(self, n_mels=80, num_blocks=6, output_size=256, embedding_dim=192, input_layer="conv2d2", 
            pos_enc_layer_type="rel_pos"):

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("num_blocks: {}".format(num_blocks))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        print("embedding_dim: {}".format(embedding_dim))
        self.specaug = FbankAug()
        self.conformer = Squeeze_ConformerEncoder(input_size=n_mels, num_blocks=num_blocks,
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
        self.pooling = AttentiveStatisticsPooling(output_size*num_blocks)
        self.bn = BatchNorm1d(input_size=output_size*num_blocks*2)
        self.fc = torch.nn.Linear(output_size*num_blocks*2, embedding_dim)
    
    def forward(self, feat, aug):
        # print("in_x", feat.shape)#([100, 1, 80, 301])
        # feat = feat.squeeze(1)  # ([100, 80, 301])
        feat = feat - torch.mean(feat, dim=-1, keepdim=True)
        if aug == True:
            feat = self.specaug(feat)
        # print("in_x1", feat.shape)#([100, 80, 301])
        feat = feat.permute(0, 2, 1)
        # print("in_x2", feat.shape)#([100, 301, 80])
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens * feat.shape[1]).int()
        # print("in_x3", feat.shape)#([100, 301, 80])
        x, masks = self.conformer(feat, lens)
        # print("in_x4", x.shape)#([100, 150, 1536])
        x = x.permute(0, 2, 1)
        # print("in_x5", x.shape)#([100, 1536, 150])
        x = self.pooling(x)
        # print("in_x6", x.shape)#([100, 3072, 1])
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        # print("out_x",x.shape)#([100, 512])
        return x

def squeeze_conformer_cat(n_mels=80, num_blocks=6, output_size=256,
        embedding_dim=192, input_layer="DW_conv2d2", pos_enc_layer_type="rel_pos"):
    model = Conformer(n_mels=n_mels, num_blocks=num_blocks, output_size=output_size, 
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model

model= squeeze_conformer_cat()
print(model)
