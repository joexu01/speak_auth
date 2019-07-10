import os
from threading import Thread

import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

from flask import current_app


def extractMFCCs(app, user_id, audio_dir, save_to):
    """
    :param app: 当前的FLASK APP
    :param user_id: 用户ID
    :param audio_dir: 声音文件的路径
    :param save_to: 将MFCC - csv 文件储存到...
    :return:
    """
    with app.app_context():
        [Fs, x] = audioBasicIO.readAudioFile(audio_dir)
        F = audioFeatureExtraction.stFeatureExtraction(x, Fs,  256, 80)
        f = F[8:21]
        np.savetxt(save_to + '/' + str(user_id) + '.csv', f, delimiter=',')
        print(f.shape)
        print('完成MFCC提取')
        os.remove(audio_dir)


def extract_mfcc(user_id, audio_dir, save_to):
    app = current_app._get_current_object()
    thr = Thread(target=extractMFCCs, args=[app, user_id, audio_dir, save_to])
    thr.start()
    return thr
