import collections
import os
import random

import numpy as np
import pandas as pd
import torch,librosa
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from .augment import WavAugment
from .FTDNNLayer import FTDNNLayer, SOrthConv
def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.

    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

def SNR(audio, snr):
    #在audio y中 添加噪声 噪声强度SNR为int
    # print("snr",snr)
    audio_power = audio ** 2
    audio_average_power = np.mean(audio_power)
    audio_average_db = 10 * np.log10(audio_average_power)
    noise_average_db = audio_average_db - snr
    noise_average_power = 10 ** (noise_average_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    return audio + noise
def load_audio(filename, second=3):
    # print("filename",filename)
    # print("second",second)
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if second <= 0:
        # waveform = waveform.astype(np.float64).copy()
        # # print("waveform", waveform.shape)
        # noise_waveform = SNR(waveform, snr=10)
        # # print("noise_waveform", noise_waveform.shape)
        # # return waveform#.copy()
        # return noise_waveform
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        # print("0")
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        # print("1")
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    waveform = np.stack([waveform], axis=0)
    return waveform#.copy()


class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.aug = aug
        # if aug:
        #     self.wav_aug = WavAugment()

        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):

        waveform_1 = load_audio(self.paths[index], self.second)
        # if self.aug:
        #     # print("1")
        #     waveform_1 = self.wav_aug(waveform_1)
        if self.pairs == False:
            # print("2")
            # print("waveform_11",waveform_1.shape)#(1, 48000)
            waveform_1 = waveform_1.squeeze()
            # print("waveform_12", waveform_1.shape)#(48000,)


            # #======================FBanks==========================
            waveform_1 = pre_emphasis(sig=waveform_1, pre_emph_coeff=0.97)
            # print("waveform_13", waveform_1.shape)#(48000,)
            S = librosa.feature.melspectrogram(y=waveform_1, sr=16000, power=1, n_fft=512, hop_length=160, n_mels=80)
            # print("waveform_14", S.shape)#(80, 301)

            logmelspec = librosa.amplitude_to_db(S)
            # print("waveform_15", logmelspec.shape)#(80, 301)

            # print("torch.FloatTensor(logmelspec)",torch.FloatTensor(logmelspec).shape)
            return torch.FloatTensor(logmelspec), self.labels[index]


        else:
            waveform_2 = load_audio(self.paths[index], self.second)
            if self.aug == True:
                waveform_2 = self.wav_aug(waveform_2)
            return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_2), self.labels[index]

    def __len__(self):
        return len(self.paths)


class Semi_Dataset(Dataset):
    def __init__(self, label_csv_path, unlabel_csv_path, second=2, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(label_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values

        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        df = pd.read_csv(unlabel_csv_path)
        self.u_paths = df["utt_paths"].values
        self.u_paths_length = len(self.u_paths)

        if label_csv_path != unlabel_csv_path:
            self.labels, self.paths = shuffle(self.labels, self.paths)
            self.u_paths = shuffle(self.u_paths)

        # self.labels = self.labels[:self.u_paths_length]
        # self.paths = self.paths[:self.u_paths_length]
        print("Semi Dataset load {} speakers".format(len(set(self.labels))))
        print("Semi Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_l = load_audio(self.paths[index], self.second)

        idx = np.random.randint(0, self.u_paths_length)
        waveform_u_1 = load_audio(self.u_paths[idx], self.second)
        if self.aug == True:
            waveform_u_1 = self.wav_aug(waveform_u_1)

        if self.pairs == False:
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1)

        else:
            waveform_u_2 = load_audio(self.u_paths[idx], self.second)
            if self.aug == True:
                waveform_u_2 = self.wav_aug(waveform_u_2)
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1), torch.FloatTensor(waveform_u_2)

    def __len__(self):
        return len(self.paths)


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        # if index>=0 and index<=4707:

        waveform = load_audio(self.paths[index], self.second)
        waveform = waveform.squeeze()
        pre_waveform = pre_emphasis(sig=waveform, pre_emph_coeff=0.97)
        S = librosa.feature.melspectrogram(y=pre_waveform, sr=16000, power=1, n_fft=512, hop_length=160, n_mels=80)

        logmelspec = librosa.amplitude_to_db(S)

        # print("torch.FloatTensor(logmelspec)",torch.FloatTensor(logmelspec).shape)
        return torch.FloatTensor(logmelspec), self.paths[index]

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    dataset = Train_Dataset(train_csv_path="data/train.csv", second=3)
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )
    for x, label in loader:
        pass

