# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
from sklearn.mixture import GaussianMixture

# python speech feature

my_file = ['1.wav', '2.wav',]

common_mix = GaussianMixture(n_components=4, covariance_type='full')
throat_mix = GaussianMixture(n_components=4, covariance_type='full')

frame_length = 256
step_length = 80

# process common data
common_data = []
for f_file in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/dataset_paper/person1/common/" + f_file)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
    f = F[8:21, ]
    f = f.T
    common_data.append(f)
common_data = np.concatenate(common_data, 0)
print(common_data.shape)
# np.savetxt(fname="common_data_online.csv", X=common_data, delimiter=',')
common_mix.fit(common_data)


# process throat data
throat_data = []
for f_file in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/dataset_paper/person2/throat/" + f_file)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
    f = F[8:21, ]
    f = f.T
    throat_data.append(f)
throat_data = np.concatenate(throat_data, 0)
print(throat_data.shape)
# np.savetxt(fname="throat_data_online.csv", X=throat_data, delimiter=',')
throat_mix.fit(throat_data)

# 预测

# [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_test/throat/2.wav")
# F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
# f = F[8:21, ].T
# print(common_mix.score(f))
# print(throat_mix.score(f))

# pre_dir = ['common', 'throat']
# pre_file = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav',
#             '6.wav', '7.wav', '8.wav', '9.wav', '10.wav',
#             '11.wav', '12.wav', '13.wav', '14.wav', '15.wav',
#             '16.wav', '17.wav', '18.wav', '19.wav', '20.wav']
#
# for p_dir in pre_dir:
#     for p_file in pre_file:
#         [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_test/" + p_dir + '/' + p_file)
#         F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#         f = F[8:21, ].T
#         common_score = common_mix.score(f)
#         throat_score = throat_mix.score(f)
#         if common_score > throat_score:
#             result = 'common'
#         else:
#             result = 'throat'
#         print(p_file, common_score, throat_score, result)

# pre_file = ['common1.wav', 'common2.wav',
#             'throat1.wav', 'throat2.wav', 'throat3.wav']
#
# for p_file in pre_file:
#     [Fs, x] = audioBasicIO.readAudioFile("D:/ML/dataset_paper/person1/" + p_file)
#     F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
#     f = F[8:21, ].T
#     common_score = common_mix.score(f)
#     throat_score = throat_mix.score(f)
#     if common_score > throat_score:
#         result = 'common'
#     else:
#         result = 'throat'
#     print(p_file, common_score, throat_score, result)

[Fs, x] = audioBasicIO.readAudioFile("D:/ML/data/test/zj-1-throat.wav")
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length, step_length)
f = F[8:21, ].T
common_score = common_mix.score(f)
throat_score = throat_mix.score(f)
if common_score > throat_score:
    result = 'common'
else:
    result = 'throat'
print(common_score, throat_score, result)
