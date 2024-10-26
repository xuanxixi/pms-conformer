# PMS-Conformer

This repository is modified based on * [MFA-Conformer](https://github.com/zyzisyz/mfa_conformer).

This repository contains the training code accompanying the paper "PMS-Conformer: Conformer-based Speaker Recognition Model for Real-time Multi-scenarios.", which is published in the journal 《Computer Engineering and Applications》 with an impact factor of **4.540**.

<p align="center"><img width="95%" src="docs/1.png" /></p>

To handle the problems of poor performances of speaker verification systems, appearing in multiple scenarios with cross-domain utterances, long-duration utterances and noisy utterances, a real-time robust speaker recognition model, PMS-Conformer, is designed based on Conformer in this paper. The architecture of the PMS-Conformer is inspired by the state-of-the-art model named MFA-Conformer. PMS-Conformer has made the improvements on the acoustic feature extractor, network components and loss calculation module of MFA-Conformer respectively, having the novel and effective acoustic feature extractor and the robust speaker embedding extractor with high generalization capability. PMS-Conformer is trained on VoxCeleb1&2 dataset, and it is compared with the baseline MFA-Conformer and ECAPA-TDNN, and extensive comparison experiments are conducted on the speaker verification tasks. The experimental results show that on VoxMovies with cross-domain utterances, SITW with long-duration utterances and VoxCeleb-O processed by adding noise to its utterances, the ASV system built with PMS-Conformer is more competitive than those built with MFA-Conformer and ECAPA-TDNN respectively.  Moreover, the trainable params and RTF of the speaker embedding extractor of PMS-Conformer are significantly lower than those of ECAPA-TDNN. All evaluation experiment results demonstrate that PMS-Conformer exhibits good performances in real-time multi- scenarios.
## System Description

The specific details can be found of my paper "Conformer-based Speaker Recognition Model for Real-time Multi-scenarios,", please check the Session 1, `PMS-Conformer`.

## Acoustic Feature Extractor

<p align="center"><img width="50%" src="docs/2.png"/></p>

## Conv_Conformer Component

<p align="center"><img width="80%" src="docs/3.png"/></p>

## Sub-center AAM-Softmax 

<p align="center"><img width="100%" src="docs/4.png"/></p>

### Dependencies

Note: That is the setting based on my device, you can modify the torch and torchaudio version based on your device.

Start from building the environment
```
conda create -n PMS python=3.7.9
source activate PMS
```
git 
```
git https://github.com/xuanxixi/pms-conformer.git
cd pms-conformer
```
Start from the existing environment
```
pip install -r requirements.txt
```

## Data Preparation

* [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
* [SITW](http://www.speech.sri.com/projects/sitw/)
* [VoxMovies](https://www.robots.ox.ac.uk/~vgg/data/voxmovies/)


```bash
# format Voxceleb test trial list
rm -rf data; mkdir data
wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
python3 scripts/format_trials.py \
            --voxceleb1_root $voxceleb1_dir \
            --src_trials_path data/veri_test.txt \
            --dst_trials_path data/vox1_test.txt

# make csv for voxceleb1&2 dev audio (train_dir)
python3 scripts/build_datalist.py \
        --extension wav \
        --dataset_dir data/$train_dir \
        --data_list_path data/train.csv
```

## Model Training

```bash
python3 main.py \
        --batch_size 200 \
        --num_workers 40 \
        --max_epochs 30 \
        --embedding_dim $embedding_dim \
        --save_dir $save_dir \
        --encoder_name $encoder_name \
        --train_csv_path $train_csv_path \
        --learning_rate 0.001 \
        --encoder_name ${encoder_name} \
        --num_classes $num_classes \
        --trial_path $trial_path \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --step_size 4 \
        --gamma 0.5 \
        --weight_decay 0.0000001 \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type 
```

## Results

### Main scenario.

<p align="center"><img width="95%" src="docs/5.png" /></p>

### Cross-domain scenario.

<p align="center"><img width="75%" src="docs/6.png"/></p>

<p align="center"><img width="75%" src="docs/7.png"/></p>

### Long-duration scenario.

<p align="center"><img width="75%" src="docs/8.png" /></p>

### Different noise scenario.

<p align="center"><img width="75%" src="docs/9.png"/></p>

### Ablation experiments.

<p align="center"><img width="75%" src="docs/10.png"/></p>

### Conclusion

In this paper, a novel real-time and multi-scenario robust automatic speaker recognition network model—PMS-Conformer—is proposed. The RTF (Real-Time Factor) of the PMS-Conformer speaker embedding extractor is 0.0172, **achieving real-time recognition performance (RTF < 0.1)**. On the primary evaluation test sets **VoxCelebO**, **SITW.Dev (core-core)**, and **SITW.Eval (core-core)**, PMS-Conformer-ASV attained EERs of **0.66%**, **1.38%**, and **1.39%**, respectively, and minDCFs of **0.0905**, **0.1388**, and **0.1347**, respectively. In cross-domain scenarios, long-duration speech scenarios, and noisy environments, PMS-Conformer-ASV demonstrated a significant advantage in EER over MFA-Conformer-ASV and ECAPA-TDNN-ASV. Ablation experiments reveal that the acoustic feature extractor Mel-MPCs and the Sub-center AAM-Softmax loss function positively contribute to ASV system performance. The Conv_Conformer module, which integrates local and global feature modeling, enhances the extraction of robust speaker embeddings, providing insights for future ASV system design. In future work, a speaker verification system based on the PMS-Conformer model will be deployed on edge computing devices to conduct real-time speaker recognition research.

## Others

What's more, here are some tips might be useful:

1. **The Conv-Conformer block**: We the borrow a lot of code from [WeNet](https://github.com/wenet-e2e/wenet) toolkit. 
2. **Average the checkpoint weights**: When the model training is done, we average the parameters of the last 3~10 checkpoints to generate a new checkpoint. The new checkpoint always tends to achieve a better recognition performance.
3. **Warmup**: We perform a linear warmup learning rate schedule at the first 2k training steps. And we find that this warmup procedure is very helpful for the model training.
4. **AS-norm**: Adaptive score normalization (AS-norm) is common trick for speaker recognition. In our experiment, it will lead to 5%-10% relative improvement in EER metric.

## Citation

If you find this code useful for your research, please cite our paper.

```
@article{xuanxixi,
  title={Conformer-based Speaker Recognition Model for Real-time Multi-scenarios},
  author={Xuan Xi and Han Runping and Gao Jingxin},
  journal={Computer Engineering and Applications},
  year={2024},
  doi={10.3778/j.issn.1002-8331.2210-0145}
}
```
```
@article{宣茜2024,
  title={基于Conformer的实时多场景说话人识别模型},
  author={宣茜, 韩润萍, 高静欣},
  journal={计算机工程与应用},
  year={2024},
  doi={10.3778/j.issn.1002-8331.2210-0145}
}
```
