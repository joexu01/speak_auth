# -*- coding: UTF-8 -*-
import numpy as np

from dtw import dtw
from numpy.linalg import norm

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

[Fs, k] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/1/1.wav")
F = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)
mfcc_1 = F[8:21]
print(mfcc_1.shape)

my_file = ['2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav', '8.wav']

for file in my_file:
    [Fs, k] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/3/" + file)
    F = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)
    mfcc_2 = F[8:21]

    dist, cost, acc_cost, path = dtw(mfcc_1.T, mfcc_2.T, dist=lambda x, y: norm(x - y, ord=1))
    print(file, ':Normalized distance between the two sounds:', dist)
