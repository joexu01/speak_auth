import numpy as np

from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction

my_dir = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
for each in my_dir:
    [Fs, x] = audioBasicIO.readAudioFile("D:/ML/speak_reg/spk_rec_data/train/" + each + "/3.wav")
    F = audioFeatureExtraction.stFeatureExtraction_modified_2nd_edition(x, Fs, 256, 80)
    F = F.T
    np.savetxt(each + 'example2.csv', F, delimiter=',')
