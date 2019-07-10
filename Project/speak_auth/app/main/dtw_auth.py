# -*- coding: UTF-8 -*-
import random
import numpy as np

from dtw import dtw
from numpy.linalg import norm

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def random_dtw_number(length=6):
    base_str = '0123456789'
    return ''.join(random.choice(base_str) for i in range(length))


# def max_list(lt):
#     temp = 0
#     for i in lt:
#         if lt.count(i) > temp:
#             max_str = i
#             temp = lt.count(i)
#     return str(max_str)
#
#
# def auth_pipeline(dtw_features_path, audio_file_path, data_matrix_path, label_matrix_path):
#     dtw_features = np.loadtxt(dtw_features_path, delimiter=',')
#
#     [Fs, k] = audioBasicIO.readAudioFile(audio_file_path)
#     audio_features = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)[8:21]
#     # audio_features = audioFeatureExtraction.stFeatureExtraction_modified_2nd_edition(k, Fs,  0.050 * Fs, 0.025 * Fs)
#
#     dist, cost, acc_cost, path = dtw(dtw_features.T, audio_features.T, dist=lambda x, y: norm(x - y, ord=1))
#     print(dist)
#     if dist < 3.0:
#         dtw_result = True
#     else:
#         dtw_result = False
#
#     data_matrix = np.loadtxt(data_matrix_path, delimiter=',')
#     label_matrix = np.loadtxt(label_matrix_path, delimiter=',')
#
#     print(data_matrix.shape, label_matrix.shape)
#
#     clf_svm = Pipeline([
#         ("scaler", StandardScaler()),
#         # ("svm_clf", SVC(kernel="poly", degree=10, C=5, coef0=100))
#         ("svm_clf", SVC(kernel='poly', gamma='scale', C=1000, decision_function_shape='ovo'))
#         # ("svm_clf", SVC(kernel='rbf', gamma='scale', C=5, decision_function_shape='ovo'))
#     ])
#
#     clf_svm.fit(data_matrix, label_matrix)
#     # svm_result = max_list(clf_svm.predict(audio_features).tolist())
#     audio_features = audioFeatureExtraction.stFeatureExtraction(k, Fs, 0.030 * Fs, 0.015 * Fs)[8:21]
#     result = clf_svm.predict(audio_features.T)
#
#     print(result.shape, result)
#
#     result_in = result.tolist()
#     counter = 0
#     for each in result_in:
#         if each == 2:
#             counter += 1
#     print(counter)
#     svm_result = max_list(result_in)
#
#     if svm_result == '2.0':
#         real_man_result = True
#     else:
#         real_man_result = False
#
#     print(dtw_result, real_man_result)
#
#     return dtw_result, real_man_result


def auth_pipeline(dtw_features_path, audio_file_path, common_data_path, throat_data_path):
    dtw_features = np.loadtxt(dtw_features_path, delimiter=',')

    [Fs, k] = audioBasicIO.readAudioFile(audio_file_path)
    audio_features = audioFeatureExtraction.stFeatureExtraction(k, Fs, 256, 80)[8:21]

    dist, cost, acc_cost, path = dtw(dtw_features.T, audio_features.T, dist=lambda x, y: norm(x - y, ord=1))
    print(dist)
    if dist < 3.0:
        dtw_result = True
    else:
        dtw_result = False

    common_data = np.loadtxt(common_data_path, delimiter=',')
    throat_data = np.loadtxt(throat_data_path, delimiter=',')

    print(common_data.shape, throat_data.shape)

    common_mix = GaussianMixture(n_components=4, covariance_type='full')
    throat_mix = GaussianMixture(n_components=4, covariance_type='full')

    common_mix.fit(common_data)
    throat_mix.fit(throat_data)

    common_score = common_mix.score(audio_features.T)
    throat_score = throat_mix.score(audio_features.T)
    if common_score > throat_score:
        real_man_result = False
    else:
        real_man_result = True

    print(common_score, throat_score, dtw_result, real_man_result)

    return dtw_result, real_man_result
