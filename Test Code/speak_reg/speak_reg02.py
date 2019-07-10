from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier


my_dir = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
          '20', '21', '22']
my_file = ['1.wav','2.wav','3.wav','4.wav','5.wav','6.wav','7.wav','8.wav']

data_matrix = []
label_matrix = []
n = 0
n = int(n)
for f_dir in my_dir:
    n += 1
    for f_file in my_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/" + f_dir + "/" + f_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
        f = F[8:21, ]
        f = f.T
        data_matrix.append(f)
        label = np.empty(f.shape[0], dtype=int)
        label = np.full(label.shape, n)
        label_matrix.append(label)
data_matrix = np.concatenate(data_matrix, 0)
label_matrix = np.concatenate(label_matrix, 0)
print(data_matrix.shape)
print(label_matrix.shape)

# gmm = GaussianMixture(n_components=4, covariance_type='full')
# gmm.fit(data_matrix, label_matrix)

# rfc = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0)
# rfc.fit(data_matrix, label_matrix)

# 预测


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return str(max_str)


pre_dir = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
           '20', '21', '22']
pre_file = ['1.wav', '2.wav']
counter = current = 0

result_matrix = []
for p_dir in pre_dir:
    # current += 1
    result_matrix.append(p_dir)
    result_matrix.append(':')
    for p_file in pre_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/test/" + p_dir + '/' + p_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
        f = F[8:21, ].T
        result = rfc.predict(f)
        result_in = result.tolist()
        result_matrix.append(max_list(result_in))
print(result_matrix)
