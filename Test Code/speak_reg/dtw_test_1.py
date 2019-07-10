# -*- coding: UTF-8 -*-
import numpy as np

from dtw import dtw
from numpy.linalg import norm
# from librosa.display import specshow
# import matplotlib as plt
from matplotlib.pyplot import imshow, plot, xlim, ylim, show, title
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

[Fs, k] = audioBasicIO.readAudioFile("D:/ML/dataset_paper/person1/throat1.wav")
F = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)


# np.savetxt("dtw1.csv", F, delimiter=',')
mfcc_1 = F[9:21]
print(mfcc_1.shape)

# my_file = ['39678_train.wav', '39678_test.wav', '39678_test2.wav', '39678_machine.wav', '40096.wav']
my_file = ['throat1.wav', 'throat2.wav', 'throat3.wav', 'common1.wav', 'common2.wav']

for file in my_file:
    [Fs, k] = audioBasicIO.readAudioFile("D:/ML/dataset_paper/person1/" + file)
    F = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)
    mfcc_2 = F[9:21]

    dist, cost, acc_cost, path = dtw(mfcc_1.T, mfcc_2.T, dist=lambda x, y: norm(x - y, ord=1))
    print(file, ':Normalized distance between the two sounds:', dist)
    imshow(cost.T, origin='lower', cmap='gray', interpolation='nearest')
    plot(path[0], path[1], 'w')
    xlim((-0.5, cost.shape[0] - 0.5))
    ylim((-0.5, cost.shape[1] - 0.5))
    title(file)

    show()
