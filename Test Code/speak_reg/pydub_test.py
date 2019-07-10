# -*- coding: UTF-8 -*-

from pydub import AudioSegment

sound = AudioSegment.from_mp3('D:/360Downloads/caixi-from-net-common.mp3').set_frame_rate(11025)
sound.export('D:/360Downloads/from-net-common-3.wav', format='wav')
