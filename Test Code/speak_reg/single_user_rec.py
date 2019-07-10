from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import numpy as np
from sklearn import svm
# python speech feature

my_dir = ['1', '2', '3', '4', '5', '6', '7']
my_file = ['1.wav', '2.wav', '3.wav', '4.wav', '5.wav', '6.wav', '7.wav', '8.wav']

data_matrix = []
label_matrix = []
for f_file in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/15/" + f_file)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ].T
    data_matrix.append(f)
    label = np.empty(f.shape[0], dtype=int)
    label = np.full(label.shape, 1)
    label_matrix.append(label)

print(np.concatenate(data_matrix, 0).shape)

for f_dir in my_dir:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/" + f_dir + "/1.wav")
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    f = f.T
    data_matrix.append(f)
    label = np.empty(f.shape[0], dtype=int)
    label = np.full(label.shape, 0)
    label_matrix.append(label)

data_matrix = np.concatenate(data_matrix, 0)
label_matrix = np.concatenate(label_matrix, 0)
print(data_matrix.shape)
print(label_matrix.shape)

# np.savetxt("feature.csv", data_matrix, delimiter=',')
# np.savetxt("label.csv", label_matrix, delimiter=',')

clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf_svm.fit(data_matrix, label_matrix)

# clf_svm = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf_svm.fit(data_matrix, label_matrix)

# 预测


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return str(max_str)


pre_dir = ['1', '2', '3', '4', '5', '6', '7', '15']
pre_file = ['1.wav', '2.wav']
counter = current = 0

result_matrix = []
for p_dir in pre_dir:
    # current += 1
    result_matrix.append(p_dir)
    result_matrix.append(':')
    for p_file in pre_file:
        [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/test/" + p_dir + '/' + p_file)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
        f = F[8:21, ].T
        result = clf_svm.predict(f)
        result_in = result.tolist()
        result_matrix.append(max_list(result_in))
print(result_matrix)
