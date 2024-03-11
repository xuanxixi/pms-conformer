#!/usr/bin/env python
# encoding: utf-8

import argparse
import os

import numpy as np
from scipy.io import wavfile
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str,
                        default="/home/xuanxi/dataset/sitw/eval/audio")
    parser.add_argument('--src_trials_path', help='src_trials_path',
                        type=str, default="/home/xuanxi/dataset/test_list2_mfa.txt")
    parser.add_argument('--dst_trials_path', help='dst_trials_path',
                        type=str, default="/home/xuanxi/dataset/VOX/vox-O-20.txt")
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)
    print("trials",trials[0])

    f = open(args.dst_trials_path, "w")
    for item in trials:
        # enroll_path = os.path.join(
        #     args.voxceleb1_root, "wav", item[1])
        # test_path = os.path.join(args.voxceleb1_root, "wav", item[2])
        # enroll_path = os.path.join(
        #     args.voxceleb1_root, item[1])
        # test_path = os.path.join(args.voxceleb1_root,  item[2])
        enroll_path = item[1]
        test_path = item[2]

        sample_rate, enroll_waveform = wavfile.read(enroll_path)
        enroll_audio_length = enroll_waveform.shape[0]

        sample_rate, test_waveform = wavfile.read(test_path)
        test_audio_length = test_waveform.shape[0]


        second=20
        length = np.int64(sample_rate * second)
        if enroll_audio_length <= length and test_audio_length <= length:
            f.write("{} {} {}\n".format(item[0], enroll_path, test_path))
        else:
            continue
