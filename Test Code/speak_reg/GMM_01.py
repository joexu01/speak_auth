from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
import io
from sklearn.mixture import gaussian_mixture

my_dir = ['A','B','C','D']
my_file = ['1.wav','2.wav','3.wav','4.wav','5.wav','6.wav','7.wav','8.wav','9.wav','10.wav','11.wav','12.wav','13.wav','14.wav','15.wav','16.wav']


def init_data(x):
    A = np.zeros(shape=(x, 1))
    B = np.zeros(shape=(x, 1))
    C = np.zeros(shape=(x, 1))
    D = np.zeros(shape=(x, 1))
    return A, B, C, D


A = np.zeros(shape=(13, 1))
B = np.zeros(shape=(13, 1))
C = np.zeros(shape=(13, 1))
D = np.zeros(shape=(13, 1))

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/A/" + ffile)
    F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    A = np.hstack((A, f))
A = A[:, 1:].T
print(A.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/B/" + ffile)
    F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    B = np.hstack((B, f))
B = B[:, 1:].T
print(B.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/C/" + ffile)
    F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    C = np.hstack((C, f))
C = C[:, 1:].T
print(C.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/D/" + ffile)
    F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:21, ]
    D = np.hstack((D, f))
D = D[:, 1:].T
print(D.shape)

A = A[:1000, ]
# np.savetxt('test01.csv', A, delimiter=',')
B = B[:1000, ]
C = C[:1000, ]
D = D[:1000, ]
# 取前1000个数据准备第一轮洗牌
shuffle_index_step1 = np.random.permutation(1000)
A = A[shuffle_index_step1]
B = B[shuffle_index_step1]
C = C[shuffle_index_step1]
D = D[shuffle_index_step1]

# REST = np.vstack((B,C,D))

# 再取洗牌后的前n_learn个数据进行学习
n_learn = 650
A = A[:n_learn, ]
# REST = REST[:1950, ]
B = B[:n_learn, ]
C = C[:n_learn, ]
D = D[:n_learn, ]

data_set = np.vstack((A,B,C,D))
data = np.mat(data_set)

A_y = np.empty(n_learn, dtype=int)
A_y = np.full(A_y.shape, 1)
B_y = np.empty(n_learn, dtype=int)
B_y = np.full(B_y.shape, 2)
C_y = np.empty(n_learn, dtype=int)
C_y = np.full(C_y.shape, 3)
D_y = np.empty(n_learn, dtype=int)
D_y = np.full(D_y.shape, 4)
label_set = np.hstack((A_y,B_y,C_y,D_y))
label = np.array(label_set)

clf = gaussian_mixture.GaussianMixture(n_components=4, covariance_type='full')
clf.fit(data, label)

# 预测
my_fflile = ['1.wav','2.wav','3.wav','4.wav']
for mydir in my_dir:
    print(mydir + '\n')
    for myfile in my_fflile:
        [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/" +
                                              mydir + "/" + myfile)
        F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 0.050 * Fs, 0.025 * Fs)
        f = F[8:21, ].T
        result = clf.predict(f)
        counter = f.shape[0]
        counter1 = counter2 = counter3 = counter4 = 0
        for i in range(0, counter):
            if result[i] == 1:
                counter1 += 1
            if result[i] == 2:
                counter2 += 1
            if result[i] == 3:
                counter3 += 1
            if result[i] == 4:
                counter4 += 1
        print(counter1, ',', counter2, ',', counter3, ',', counter4, '\n')
