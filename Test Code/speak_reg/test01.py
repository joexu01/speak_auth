from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import io


my_dir = ['A','B','C','D']
my_file = ['1.wav','2.wav','3.wav','4.wav','5.wav','6.wav','7.wav','8.wav','9.wav','10.wav','11.wav','12.wav','13.wav','14.wav','15.wav','16.wav']
# A = np.zeros(shape=(12, 1))
# B = np.zeros(shape=(12, 1))             ,'11.wav','12.wav','13.wav','14.wav','15.wav','16.wav'
# C = np.zeros(shape=(12, 1))
# D = np.zeros(shape=(12, 1))
# for ffile in my_file:
#     [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/A/" + ffile)
#     F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#     f = F[8:20, ]
#     A = np.hstack((A,f))
# A = A[:, 1:]
#
#
# for ffile in my_file:
#     [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/B/" + ffile)
#     F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#     f = F[8:20, ]
#     B = np.hstack((B,f))
# B = B[:, 1:]
#
#
# for ffile in my_file:
#     [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/C/" + ffile)
#     F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#     f = F[8:20, ]
#     C = np.hstack((C,f))
# C = C[:, 1:]
#
#
# for ffile in my_file:
#     [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/D/" + ffile)
#     F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
#     f = F[8:20, ]
#     D = np.hstack((D,f))
# D = D[:, 1:]

A = np.zeros(shape=(12, 1))
B = np.zeros(shape=(12, 1))
C = np.zeros(shape=(12, 1))
D = np.zeros(shape=(12, 1))

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/A/" + ffile)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:20, ]
    A = np.hstack((A,f[:,10:80]))
A = A[:, 1:].T
print(A.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/B/" + ffile)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:20, ]
    B = np.hstack((B,f[:,10:80]))
B = B[:, 1:].T
print(B.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/C/" + ffile)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:20, ]
    C = np.hstack((C,f[:,10:80]))
C = C[:, 1:].T
print(C.shape)

for ffile in my_file:
    [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/train/D/" + ffile)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    f = F[8:20, ]
    D = np.hstack((D, f[:,10:80]))
D = D[:, 1:].T
print(D.shape)

A = A[:1000, ]
B = B[:1000, ]
C = C[:1000, ]
D = D[:1000, ]
# 取前1000个数据准备第一轮洗牌
shuffle_index_step1 = np.random.permutation(1000)
A = A[shuffle_index_step1]
B = B[shuffle_index_step1]
C = C[shuffle_index_step1]
D = D[shuffle_index_step1]

# 再取洗牌后的前n_learn个数据进行学习
n_learn = 650
A = A[:n_learn, ]
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

# 对训练数据进行第二次洗牌
shuffle_index_step2 = np.random.permutation(2400)
data = data[shuffle_index_step2]
label = label[shuffle_index_step2]
# print(data.shape)
# print(label.shape)


# 支持向量机
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(data, label)

# 随机森林
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf.fit(data, label)


# output = np.hstack((A,B,C,D))
# output = A
# np.savetxt('datamixed.csv', output, delimiter=',')

# 预测
my_dir = ['A','B','C','D']
my_fflile = ['1.wav','2.wav','3.wav','4.wav']
for mydir in my_dir:
    print(mydir + '\n')
    for myfile in my_fflile:
        [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/" +
                                              mydir + "/" + myfile)
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
        f = F[8:20, ].T
        result = clf.predict(f[35:80])
        counter1 = counter2 = counter3 = counter4 = 0
        for i in range(0, 45):
            if result[i] == 1:
                counter1 += 1
            if result[i] == 2:
                counter2 += 1
            if result[i] == 3:
                counter3 += 1
            if result[i] == 4:
                counter4 += 1
        print(counter1, ',', counter2, ',', counter3, ',', counter4, '\n')

# [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/D/4.wav")
# F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
# f = F[8:20, ].T
# print(f.shape)
# result = clf.predict(f[10:40])
#
# counter1 = counter2 = counter3 = counter4 = 0
# for i in range(0,30):
#     if result[i]==1:
#         counter1 += 1
#     if result[i]==2:
#         counter2 += 1
#     if result[i]==3:
#         counter3 += 1
#     if result[i]==4:
#         counter4 += 1
# print(counter1,',',counter2,',',counter3,',',counter4)

# 交叉验证
print(cross_val_score(clf, data, label, cv=3, scoring="accuracy"))

# result_plot = {'A':counter1,'B':counter2,'C':counter3,'D':counter4}
# names = {'A','B','C','D'}
# values = {counter1,counter2,counter3,counter4}
#
#
# fig, axs = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
# axs[0].bar(names, values)
# fig.suptitle('预测结果')
# plt.show()
# print('\n')
# print(F)  ~PycharmProjects/AudioProcessing/
# f = F[0, ]
# print(f)
# output = f.T
# np.savetxt('data.csv', F, delimiter=' ')
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
