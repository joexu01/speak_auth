# -*- coding: UTF-8 -*-

def stFeatureExtraction_modified(signal, fs, win, step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        fs:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
    RETURNS
        st_features:   a numpy array (n_feats x numOfShortTermWindows)
    """

    win = int(win)
    step = int(step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(win / 2)

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats
#    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
#     feature_names = []
#     feature_names.append("zcr")
#     feature_names.append("energy")
#     feature_names.append("energy_entropy")
#     feature_names += ["spectral_centroid", "spectral_spread"]
#     feature_names.append("spectral_entropy")
#     feature_names.append("spectral_flux")
#     feature_names.append("spectral_rolloff")
#     feature_names += ["mfcc_{0:d}".format(mfcc_i)
#                       for mfcc_i in range(1, n_mfcc_feats+1)]
#     feature_names += ["chroma_{0:d}".format(chroma_i)
#                       for chroma_i in range(1, n_chroma_feats)]
#     feature_names.append("chroma_std")
    st_features = []
    while (cur_p + win - 1 < N):                        # for each short-term window until the end of signal
        count_fr += 1
        x = signal[cur_p:cur_p+win]                    # get current window
        cur_p = cur_p + step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy()                             # keep previous fft mag (used in spectral flux)
        if stEnergy(x) <= 0.01:
            continue
        curFV = numpy.zeros((n_total_feats, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, X_prev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, fs)        # spectral rolloff
        curFV[n_time_spectral_feats:n_time_spectral_feats+n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
        chromaNames, chromaF = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
        curFV[n_time_spectral_feats + n_mfcc_feats:
              n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF
        curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF.std()
        st_features.append(curFV)
        # delta features
        '''
        if count_fr>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        st_features.append(curFVFinal)        
        '''
        # end of delta
        X_prev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features


def stFeatureExtraction_modified_2nd_edition(signal, fs, win, step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        fs:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
    RETURNS
        st_features:   a numpy array (n_feats x numOfShortTermWindows)
    """

    win = int(win)
    step = int(step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(win / 2)

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)

    # n_time_spectral_feats = 8
    # n_harmonic_feats = 0
    n_mfcc_feats = 13
#     n_chroma_feats = 13
#     n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats
# #    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
#     feature_names = []
#     feature_names.append("zcr")
#     feature_names.append("energy")
#     feature_names.append("energy_entropy")
#     feature_names += ["spectral_centroid", "spectral_spread"]
#     feature_names.append("spectral_entropy")
#     feature_names.append("spectral_flux")
#     feature_names.append("spectral_rolloff")
#     feature_names += ["mfcc_{0:d}".format(mfcc_i)
#                       for mfcc_i in range(1, n_mfcc_feats+1)]
#     feature_names += ["chroma_{0:d}".format(chroma_i)
#                       for chroma_i in range(1, n_chroma_feats)]
#     feature_names.append("chroma_std")
    st_features = []
    while (cur_p + win - 1 < N):                        # for each short-term window until the end of signal
        count_fr += 1
        x = signal[cur_p:cur_p+win]                    # get current window
        cur_p = cur_p + step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy()                             # keep previous fft mag (used in spectral flux)
        if stEnergy(x) < 0.01:
            continue
        curFV = numpy.zeros((n_mfcc_feats, 1))
        # curFV[0] = stZCR(x)                              # zero crossing rate
        # curFV[1] = stEnergy(x)                           # short-term energy
        # curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        # [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
        # curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        # curFV[6] = stSpectralFlux(X, X_prev)              # spectral flux
        # curFV[7] = stSpectralRollOff(X, 0.90, fs)        # spectral rolloff
        curFV[0:n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
        # chromaNames, chromaF = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
        # curFV[n_time_spectral_feats + n_mfcc_feats:
        #       n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
        #     chromaF
        # curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
        #     chromaF.std()
        st_features.append(curFV)
        # delta features
        '''
        if count_fr>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        st_features.append(curFVFinal)        
        '''
        # end of delta
        X_prev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features