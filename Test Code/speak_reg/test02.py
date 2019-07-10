# [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/A/1.wav");
# F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs);
# f1 = F[8:20, ]
#
# [Fs, x] = audioBasicIO.readAudioFile("/home/joexu01/PycharmProjects/speak_reg/data/test/A/2.wav");
# F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050 * Fs, 0.025 * Fs);
# f2 = F[8:20, ]
# print(f2.shape)
#
# output = np.vstack((f1, f2))
# np.savetxt('test02.csv', output, delimiter=',')  sa1.wav

# x = np.zeros((6,6), np.char)
# x = np.full(x.shape, 'A')q

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STXihei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
# font = {
#     'family': 'Microsoft YaHei',
#     'weight': 'light',
# }
# plt.rc('font', **font)

[Fs, x] = audioBasicIO.readAudioFile("D:/AudioProcessing/train.wav")
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 256, 80)
f = F[9:21, ].T
print(f.shape)
plt.subplot(2, 1, 1); plt.plot(F[1, :]); plt.xlabel('帧'); plt.ylabel('短时能量');
plt.subplot(2, 1, 2); plt.plot(f); plt.xlabel('帧'); plt.ylabel('MFCCs'); plt.show();


# F = audioFeatureExtraction.stFeatureExtraction_modified(x, Fs, 256, 80)
# f = F[8:21, ].T
# print(f.shape)
# plt.subplot(1, 1, 1); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show();