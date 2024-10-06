from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Union
import torch.distributed as dist
from pytorch_lightning.plugins import DDPPlugin
import random

import torch
import torch.nn as nn
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CyclicLR

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
import score as score
from loss import softmax, amsoftmax
from loss.amsoftmax import AAMsoftmax
from loss.SubcenterArcMarginProduct import SubcenterArcMarginProduct
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")
score_file = open('/home/xxuan/pms-conformer-2024/exps1211/exps4/score.txt', "a+")
# print("score_file",score_file)
class Task(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 10,
        max_epochs: int = 1000,
        trial_path: str = "data/vox1_test.txt",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.mel_trans = Mel_Spectrogram()

        from module.resnet import resnet34, resnet18, resnet34_large
        from module.ecapa_tdnn import ecapa_tdnn, ecapa_tdnn_large
        from module.transformer_cat import transformer_cat
        from module.conformer import conformer
        from module.conformer_cat import conformer_cat
        from module.new_conformer_cat import new_conformer_cat
        from module.squeeze_conformer_cat import squeeze_conformer_cat
        from module.CNTF_conformer_cat import CNTF_conformer_cat
        from module.conformer_weight import conformer_weight


        if self.hparams.encoder_name == "resnet18":
            self.encoder = resnet18(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "resnet34":
            self.encoder = resnet34_large(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "ecapa_tdnn_large":
            self.encoder = ecapa_tdnn_large(embedding_dim=self.hparams.embedding_dim)

        elif self.hparams.encoder_name == "conformer":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)

        elif self.hparams.encoder_name == "transformer_cat":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = transformer_cat(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)


        elif self.hparams.encoder_name == "CNTF_conformer_cat":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = CNTF_conformer_cat(embedding_dim=self.hparams.embedding_dim,
                    num_blocks=self.hparams.num_blocks, input_layer="CNTF",
                    pos_enc_layer_type=self.hparams.pos_enc_layer_type)

        elif self.hparams.encoder_name == "conformer_weight":
            print("num_blocks is {}".format(self.hparams.num_blocks))
            self.encoder = conformer_weight(embedding_dim=self.hparams.embedding_dim, 
                    num_blocks=self.hparams.num_blocks, input_layer=self.hparams.input_layer)

        else:
            raise ValueError("encoder name error")

        if self.hparams.loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)

        elif self.hparams.loss_name == "sc-aamsoftmax":
            self.loss_fun = SubcenterArcMarginProduct(in_features=self.hparams.embedding_dim,
                                                      out_features=self.hparams.num_classes,
                                                      K=2,
                                                      s=30.0,
                                                      m=0.20,
                                                      easy_margin=False)
        elif self.hparams.loss_name == "AAMsoftmax":
            self.loss_fun = AAMsoftmax(n_class = self.hparams.num_classes, m = 0.20, s = 30.0)
        else:
            self.loss_fun = softmax(embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)

        self.epoch = 0
    def forward(self, x):
        # feature = self.mel_trans(x)
        embedding = self.encoder(x, aug=True)
        return embedding

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        # feature = self.mel_trans(waveform)
        embedding = self.encoder(waveform, aug=True)
        loss, acc = self.loss_fun(embedding, label)
        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            # x = self.mel_trans(x)
            self.encoder.eval()
            x = self.encoder(x, aug=False)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        num_gpus = 1#torch.cuda.device_count()
        eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(eval_vectors, self.eval_vectors)
        eval_vectors = np.vstack(eval_vectors)

        table = [None for _ in range(num_gpus)]
        dist.all_gather_object(table, self.index_mapping)

        index_mapping = {}
        for i in table:
            index_mapping.update(i)

        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)
        labels, scores = score.cosine_score(
            self.trials, index_mapping, eval_vectors)
        print("scores",scores)
        # scores = score.cosine_score(
        #     self.trials, index_mapping, eval_vectors)
        from score.cosine import asnorm
        # scores = "/home/xuanxi/mfa/mfa_conformer-master/exps/exps21/asnorm-VOX/test_list2_mfa_save.txt"
        # cohort = "/home/xuanxi/mfa/mfa_conformer-master/exps/exps21/asnorm-VOX/enroll_cohort.txt"
        # scores = asnorm(scores, cohort, nTop= 300)
        # np.savetxt('/home/xuanxi/mfa_conformer-master/exps/exps3/Vox/labels-O.txt', labels)
        # np.savetxt('/home/xuanxi/mfa_conformer-master/exps/exps3/Vox/scores-O.txt', scores)
        # np.savetxt('/home/xuanxi/mfa/mfa_conformer-master/exps/exps2/Vox/labels5.txt', labels)
        # np.savetxt('/home/xuanxi/mfa/mfa_conformer-master/exps/exps2/Vox/scores5.txt', scores)
        # np.savetxt('./exps/exps25/Vox-movies/labels-e1.txt', labels)
        # np.savetxt('./exps/exps25/Vox-movies/scores-e1.txt', labels)
        # np.savetxt('./exps/exp51/Vox-movies/scores-pooled1+5.txt', scores)
        # np.savetxt('./exps/exp61/Vox-movies/labels-pooled1-5.txt', labels)
        # np.savetxt('./exps/exp61/Vox-movies/scores-pooled1-5.txt', scores)
        # np.savetxt('./exps/exps26/Vox/labels-sitw-dev.txt', labels)#dev
        # np.savetxt('./exps/exps26/Vox/scores-sitw-dev.txt', scores)#eval
        # np.savetxt('./exps/exps21/Vox-sec/labels-6.txt', labels)
        # np.savetxt('./exps/exps21/Vox-sec/scores-6.txt', scores)
        # np.savetxt('./exps/exps23/as-norm-small/as-morm-success/big/enroll_cohort_score.txt', scores)
        # np.savetxt('./exps/exps23/as-morm-success/big/test_cohort_score.txt', scores)
        # np.savetxt('./exps/exps21/VoxSNR/labels-O-10.txt', labels)
        # np.savetxt('./exps/exps21/VoxSNR/scores-O-10.txt', scores)
        # np.savetxt('./exps/exps23/sitw/labels-30.txt', labels)
        # np.savetxt('./exps/exps23/sitw/scores-30.txt', scores)
        EER, threshold = score.compute_eer(labels, scores)

        # print("\n===============Epoch {%d}================\n" % self.epoch)
        print("===Test experimental results=======")

        print("\nEER= {:.6f}% ".format(EER * 100))
        print("\nthreshold= {:.6f}% ".format(threshold))

        self.log("cosine_eer", EER * 100)

        minDCF1, threshold1 = score.compute_minDCF(labels, scores, p_target=0.1)
        print("cosine minDCF(10-1): {:.6f} with threshold {:.2f}".format(minDCF1, threshold1))
        self.log("cosine_minDCF(10-1)", minDCF1)

        minDCF2, threshold2 = score.compute_minDCF(labels, scores, p_target=0.01)
        print("minDCF(Ptarget=0.01)= {:.6f} ".format(minDCF2))
        print("\nthreshold2= {:.6f}% ".format(threshold2))
        self.log("cosine_minDCF(10-2)", minDCF2)

        minDCF3, threshold3 = score.compute_minDCF(labels, scores, p_target=0.001)
        print("cosine minDCF(10-3): {:.6f} with threshold {:.2f}".format(minDCF3, threshold3))
        self.log("cosine_minDCF(10-3)", minDCF3)

        print("====================================")
        # exit()
        score_file.write("===============Epoch {%d}================\n" % self.epoch)
        score_file.write("cosine EER: {:.6f}% with threshold {:.6f}\n".format(EER * 100, threshold))
        score_file.write("cosine minDCF(10-1): {:.6f} with threshold {:.6f}\n".format(minDCF1, threshold1))
        score_file.write("cosine minDCF(10-2): {:.6f} with threshold {:.6f}\n".format(minDCF2, threshold2))
        score_file.write("cosine minDCF(10-3): {:.6f} with threshold {:.6f}\n".format(minDCF3, threshold3))
        score_file.flush()
        # exit()
        self.epoch += 1


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up learning_rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / float(self.hparams.warmup_step))
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = lr_scale * self.hparams.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        (args, _) = parser.parse_known_args()

        parser.add_argument("--num_workers", default=28, type=int)
        parser.add_argument("--embedding_dim", default=512, type=int)
        # parser.add_argument("--num_classes", type=int, default=7205)
        parser.add_argument("--num_classes", type=int, default=1211)
        parser.add_argument("--num_blocks", type=int, default=6)

        parser.add_argument("--input_layer", type=str, default="conv2d2")#conv2d2
        parser.add_argument("--pos_enc_layer_type", type=str, default="rel_pos")

        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=200)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=2000)
        parser.add_argument("--weight_decay", type=float, default=0.000001)

        parser.add_argument("--save_dir", type=str,
                            # default="/home/xxuan/pms-conformer-2024/exps1211/exps1")#pms-conformer
                            # default="/home/xxuan/pms-conformer-2024/exps1211/exps2")  # ecapa_tdnn+fbanks
                            # default="/home/xxuan/pms-conformer-2024/exps1211/exps3")  # mfa-conformer
                            default="/home/xxuan/pms-conformer-2024/exps1211/exps4")  # pms-conformer 15-->3
        parser.add_argument("--checkpoint_path",
                            type=str,
                            default=None)
                            # default = "/home/xxuan/pms-conformer-2023/exps/mfa/epoch=15_cosine_eer=0.65.ckpt")
                            # default="/home/xxuan/pms-conformer-2023/exps/pms/epoch=37_cosine_eer=0.66.ckpt")
                            # default = "/home/xuanxi/mfa/mfa_conformer-master/exps/exps21/epoch=37_cosine_eer=0.66.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps20/epoch=8_cosine_eer=0.86.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps2/epoch=11_cosine_eer=0.96.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps15/epoch=19_cosine_eer=0.82.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps17/epoch=12_cosine_eer=0.85.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps3/epoch=9_cosine_eer=0.83.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps8/epoch=18_cosine_eer=0.61.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps6/epoch=17_cosine_eer=0.79.ckpt")
                            # default = "/home/xuanxi/mfa/mfa_conformer-master/exps/exps18/epoch=11_cosine_eer=0.79.ckpt")
                            # default = "/home/xuanxi/mfa/mfa_conformer-master/exps/exps2/epoch=7_cosine_eer=0.98.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps27/epoch=18_cosine_eer=0.72.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps23/epoch=15_cosine_eer=0.65.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps25/epoch=23_cosine_eer=0.66.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps26/epoch=19_cosine_eer=0.65.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps24/epoch=14_cosine_eer=0.91.ckpt")
                            # default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps21/epoch=37_cosine_eer=0.66.ckpt")
        #"/home/xuanxi/文档/mfa_conformer-master/exps/exps1/epoch=3_cosine_eer=1.52.ckpt"
        parser.add_argument("--loss_name", type=str, default="sc-aamsoftmax")#amsoftmax
        parser.add_argument("--encoder_name", type=str, default="conformer")
        # parser.add_argument("--encoder_name", type=str, default="ecapa_tdnn")

        # parser.add_argument("--train_csv_path", type=str, default="/home/xxuan/dataset/list/train_list_7205_cityu.csv")
        parser.add_argument("--train_csv_path", type=str, default="/home/xxuan/dataset/list/vox1-1211-trainlist.csv")
        parser.add_argument("--trial_path", type=str, default="/home/xxuan/pms-conformer-2024/list/zhuti/voxceleb-O.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xxuan/pms-conformer-2023/list/zhuti/eval-core.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xxuan/pms-conformer-2023/list/zhuti/1.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/test_list2_mfa.txt")
        # parser.add_argument("--trial_path", type=str,
        #                     default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps21/as-norm-small/as-morm-success/big/enroll_cohort_eval.txt")
        # parser.add_argument("--trial_path", type=str,
        #                     default="/home/xuanxi/mfa/mfa_conformer-master/exps/exps23/as-morm-success/big/test_cohort_eval.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/Vox-H-clean_mfa.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/Vox-E-clean_mfa.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/sitw/all-core-30.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/sitw/dev-core.txt")#dev;eval
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/Vox-E-clean_mfa.txt")
        # parser.add_argument("--trial_path", type=str,
        #                     default="/home/xuanxi/dataset/vox_movies/evaluation_sets/e1.txt")
        # parser.add_argument("--trial_path", type=str, default="/home/xuanxi/dataset/VOX/vox-O-6.txt")
        # parser.add_argument("--trial_path", type=str,
        #                     default="/home/xuanxi/dataset/vox_movies/evaluation_sets/pooled30.txt")
        parser.add_argument("--score_save_path", type=str, default=None)

        parser.add_argument('--eval', action='store_true', default="False")
        parser.add_argument('--aug', action='store_true', default="False")
        return parser


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Task(**args.__dict__)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
        # print("state_dict",state_dict)
        model.load_state_dict(state_dict, strict=True)
        print("load weight from {}".format(args.checkpoint_path))

    assert args.save_dir is not None
    checkpoint_callback = ModelCheckpoint(monitor='cosine_eer', save_top_k=100,
           filename="{epoch}_{cosine_eer:.2f}", dirpath=args.save_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # init default datamodule
    print("data augmentation {}".format(args.aug))
    dm = SPK_datamodule(train_csv_path=args.train_csv_path, trial_path=args.trial_path, second=args.second,
            aug=args.aug, batch_size=args.batch_size, num_workers=args.num_workers, pairs=False)
    AVAIL_GPUS = torch.cuda.device_count()
    print("AVAIL_GPUS",AVAIL_GPUS)
    trainer = Trainer(
            max_epochs=args.max_epochs,
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=AVAIL_GPUS,
            num_sanity_val_steps=-1,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback, lr_monitor],
            default_root_dir=args.save_dir,
            # reload_dataloaders_every_n_epochs=1,
            accumulate_grad_batches=1,
            log_every_n_steps=25,
            )
    if args.eval:
        trainer.test(model, datamodule=dm)
    else:
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()

