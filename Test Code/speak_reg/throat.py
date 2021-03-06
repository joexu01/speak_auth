# -*- coding: UTF-8 -*-

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# python speech feature

my_dir = ['common', 'throat']
my_file = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav', '8.wav', '9.wav', '10.wav',
           'from-net-1.wav',
           'from-net-2.wav', 'from-net-3.wav'
           ]
# my_file = ['from-net-1.wav',
#            'from-net-2.wav', 'from-net-3.wav']
# my_file = ['train.wav']

frame_length = 0.07
step_length = 0.035

data_matrix = []
label_matrix = []
n = 0
n = int(n)
for f_dir in my_dir:
    n += 1
    for f_file in my_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_train/" + f_dir + "/" + f_file)
        # F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length * Fs, step_length * Fs)
        f = F[8:21, ]
        f = f.T
        data_matrix.append(f)
        label = np.empty(f.shape[0], dtype=int)
        label = np.full(label.shape, n)
        label_matrix.append(label)
data_matrix = np.concatenate(data_matrix, 0)
label_matrix = np.concatenate(label_matrix, 0)

# data_matrix = data_matrix
# label_matrix = label_matrix
print(data_matrix.shape)
print(label_matrix.shape)

# data_matrix_new = SelectKBest(f_regression, k=9).fit_transform(data_matrix, label_matrix)

np.savetxt(fname="data_matrix.csv", X=data_matrix, delimiter=',')
np.savetxt("label_matrix.csv", label_matrix, delimiter=',')

# clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
# clf_svm.fit(data_matrix, label_matrix)

clf_svm = Pipeline([
    ("scaler", StandardScaler()),
    # ("svm_clf", SVC(kernel="poly", C=1000))
    # ("svm_clf", SVC(kernel='rbf', gamma='scale', C=5, decision_function_shape='ovo'))
    ("svm_clf", SVC(kernel='poly', gamma='scale', C=100, decision_function_shape='ovo'))
    # ("gmm_clf", GaussianMixture(n_components=2, covariance_type='full'))
])

# clf_svm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_svm.fit(data_matrix, label_matrix)


# RandomForest

# 预测


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return str(max_str)


# pre_dir = ['common', 'throat']
# pre_file = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav', '8.wav', '9.wav',
#             '10.wav', '11.wav', '12.wav', '13.wav', '14.wav', '15.wav', 'xsb-common.wav.wav', ]
# counter = current = 0
#
# result_matrix = []
# for p_dir in pre_dir:
#     # current += 1
#     result_matrix.append(p_dir)
#     result_matrix.append(':')
#     for p_file in pre_file:
#         [Fs, x] = audioBasicIO.readAudioFile("D:/ML/mono_test/" + p_dir + '/' + p_file)
#         F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#         # F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
#         f = F[8:21, ].T
#         result = clf_svm.predict(f)
#         result_in = result.tolist()
#         result_matrix.append(max_list(result_in))
# print(result_matrix)

pre_file = ['common1.wav', 'common2.wav', 'common3.wav', 'common4.wav', 'common5.wav',
            'common6.wav', 'common7.wav', 'common8.wav', 'common9.wav', 'common10.wav',
            't1.wav', 't2.wav', 't3.wav', 't4.wav', 't5.wav',
            't6.wav', 't7.wav', 't8.wav', 't9.wav', 't10.wav',
            't11.wav', 't12.wav', 't13.wav', 't14.wav', 't15.wav']

result_matrix = []
for p_file in pre_file:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/data/SVM/" + p_file)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_length * Fs, step_length * Fs)
    # F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
    f = F[8:21, ].T
    result = clf_svm.predict(f)
    result_in = result.tolist()
    # result_matrix.append(p_file)
    result_matrix.append(max_list(result_in))
print(result_matrix)
