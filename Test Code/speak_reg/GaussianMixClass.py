# -*- coding: UTF-8 -*-
from pyAudioAnalysis import audioFeatureExtraction, audioBasicIO
from sklearn.mixture import GaussianMixture
import numpy as np


class RealManGaussianMixtureClass:
    common_mix = GaussianMixture(n_components=4, covariance_type='full')
    throat_mix = GaussianMixture(n_components=4, covariance_type='full')

    def __init__(self, common_data, throat_data):
        super(RealManGaussianMixtureClass, self).__init__()
        self.common_mix.fit(common_data)
        self.throat_mix.fit(throat_data)
        print("Initialized Successfully!")

    def score_a_sample(self, file_path):
        [sample_rate, signal] = audioBasicIO.readAudioFile(file_path)
        audio_features = audioFeatureExtraction.stFeatureExtraction(
            signal=signal, fs=sample_rate, win=256, step=80)[8:21, ].T
        common_score = self.common_mix.score(audio_features)
        throat_score = self.throat_mix.score(audio_features)
        if common_score > throat_score:
            result = 'common'
        else:
            result = 'throat'
        print(file_path.split('/')[-1], common_score, throat_score, result)


if __name__ == '__main__':
    my_common_file = ['common-1.wav', 'common-2.wav', 'common-3.wav',
                      'common-4.wav', 'common-5.wav', 'common-6.wav']
    # , 'common-4.wav', 'common-5.wav', 'common-6.wav'
    my_throat_file = ['throat-1.wav', 'throat-2.wav', 'throat-3.wav',
                      'throat-4.wav', 'throat-5.wav', 'throat-6.wav']
    # , 'throat-4.wav', 'throat-5.wav', 'throat-6.wav'
    frame_length = 256
    step_length = 80

    # process common data
    common_data_input = []
    for f_file in my_common_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_train/common/" + f_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
        f = F[8:21, ]
        f = f.T
        common_data_input.append(f)
    common_data_input = np.concatenate(common_data_input, 0)
    print(common_data_input.shape)

    # process throat data
    throat_data_input = []
    for f_file in my_throat_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_train/throat/" + f_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
        f = F[8:21, ]
        f = f.T
        throat_data_input.append(f)
    throat_data_input = np.concatenate(throat_data_input, 0)
    print(throat_data_input.shape)

    real_test = RealManGaussianMixtureClass(common_data=common_data_input, throat_data=throat_data_input)

    real_test.score_a_sample("D:/ML/data/SVM/common1.wav")
