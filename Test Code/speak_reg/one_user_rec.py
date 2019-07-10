from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture
import os

RATE = float(0.75)

# 提取特征
my_file = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav', '8.wav']

person = '12'
data_matrix = []
label_matrix = []

for file in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/" + person + '/' + file)
    F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    f = f.T
    data_matrix.append(f)
    label = np.empty(f.shape[0], dtype=int)
    label = np.full(label.shape, int(person))
    label_matrix.append(label)
data_matrix = np.concatenate(data_matrix, 0)
label_matrix = np.concatenate(label_matrix, 0)

print(data_matrix.shape)
print(label_matrix.shape)

# clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
# clf_svm.fit(data_matrix, label_matrix)
gmm = GaussianMixture(n_components=1, covariance_type='full')
gmm.fit(data_matrix, label_matrix)


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return str(max_str)


# lt->list, lb->label
def calculate_rate(lt, total, lb):
    counter = 0
    for item in lt:
        if item == lb:
            counter += 1
    return float(counter / total)


# 预测
pre_dir = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
           '20', '21', '22']
pre_file = ['1.wav', '2.wav']

# result_str = ''
for p_dir in pre_dir:
    print(p_dir + ': '),
    for p_file in pre_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/test/" + p_dir + '/' + p_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
        f = F[8:21, ].T
        result = gmm.predict(f)
        # if calculate_rate(result.tolist(), float(result.shape[0]), p_dir) >= RATE:
        #     print('Yes '),
        # else:
        #     print('No'),
        print(result)