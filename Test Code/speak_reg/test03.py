import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn import svm
from os import listdir
from random import sample


def load_from_other_csvs_refreshed(file_dir, current_filename, tag):
    """
    :param file_dir: 存放 mfcc 参数的csv 文件的路径
    :param current_filename: 当前用户的 mfcc 参数的csv 文件名
    :param tag: 标记其他用户数据为0
    :return: 返回其他用户的 mfcc 参数矩阵，以及用做标记的标签列表
    """
    all_files = listdir(file_dir)  # 将所有文件名存入一个 list
    all_files = sample(all_files, 16)  # 随机提取6个文件名
    if current_filename in all_files:
        all_files.remove(current_filename)  # 如果当前用户的mfcc文件在list中，就把它从list中删除
    else:
        del all_files[-1]  # 否则删除列表中最后一个文件名，确保提取出来的是5个人的mfcc参数
    mfccs_matrix = []
    label_matrix = []
    for file in all_files:
        mfccs = np.loadtxt(file_dir + '/' + file, delimiter=',')  # 加载csv文件
        mfccs_matrix.append(mfccs)
        label = np.empty(mfccs.shape[0], dtype=int)
        label = np.full(label.shape, int(tag))
        label_matrix.append(label)
    mfccs_matrix = np.concatenate(mfccs_matrix, 0)
    label_matrix = np.concatenate(label_matrix, 0)
    # 最后对五个数据进行洗牌，确保每个人的数据都混杂在一起
    # mfccs_matrix = np.random.permutation(mfccs_matrix)
    print(mfccs_matrix.shape, label_matrix.shape)
    return mfccs_matrix, label_matrix


def mfcc_auth(data_matrix, label_matrix, predict_matrix):
    clf_svm = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf_svm.fit(data_matrix, label_matrix)
    result = clf_svm.predict(predict_matrix).tolist()
    print(result)
    counter = 0
    for each in result:
        if each == 1:
            counter += 1
    percentage = counter / len(result)
    print(percentage)
    if percentage >= 0.50:
        return True
    else:
        return False


def load_mfccs_from_csv(csv_path, tag):
    mfccs = np.loadtxt(csv_path, delimiter=',')
    label = np.empty(mfccs.shape[0], dtype=int)
    label = np.full(label.shape, tag)
    return mfccs, label


def extract_mfcc_not_threading(audio_path):
    [Fs, x] = audioBasicIO.readAudioFile(audio_path)
    F = audioFeatureExtraction.stFeatureExtraction_modified_2nd_edition(x, Fs, 256, 80)
    mfccs = F.T
    return mfccs


test_data = extract_mfcc_not_threading('D:/AudioProcessing/test.wav')  # 测试数据集

train_data_1 = extract_mfcc_not_threading('D:/AudioProcessing/train.wav')  # 训练数据集
label_1 = np.empty(train_data_1.shape[0], dtype=int)
label_1 = np.full(train_data_1.shape, 1)


