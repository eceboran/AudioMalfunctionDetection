import os
import numpy as np
import pandas as pd
from utils.get_mel_spectrogram import get_mel_spectrogram


def get_melspec_features_from_files(data_dir, df_input, window, overlap = 0.5, n_fft = None, n_mels = 32,
                                    fmin = 0, fmax = None, no_channel = None, normalize_signal = False, feature_type='mel_spect_db'):

    X = []
    y = []

    for idx, row in df_input.iterrows():
        file_path = os.path.join(data_dir, row.file_rel_path)
        fs = row.fs_Hz

        mel_spect, mel_spect_db, mfcc, params = get_mel_spectrogram(file_path, 
                                                                    window=window, 
                                                                    overlap=overlap,
                                                                    n_fft=n_fft,
                                                                    n_mels=n_mels,
                                                                    fmin=fmin,
                                                                    fmax=fmax,
                                                                    no_channel=no_channel,
                                                                    machine=row.machine,
                                                                    normalize_signal=normalize_signal)
        
        if feature_type=='mel_spect':
            sample_features = mel_spect.flatten().reshape(-1, 1)
        elif feature_type=='mel_spect_db':
            sample_features = mel_spect_db.flatten().reshape(-1, 1)
        elif feature_type == 'mfcc':
            sample_features = mfcc.flatten().reshape(-1, 1)

        ft_grid_names = params['ft_grid_names'].flatten().reshape(-1, 1)
            
        sample_target = row.anomaly
        X.append(sample_features)
        y.append(sample_target)
        
    X = np.array(X)
    X = X[:,:,0]
    y = np.array(y).reshape(-1, 1)

    X = pd.DataFrame(X)
    X.columns = ft_grid_names.flatten()
    y = pd.DataFrame(y)
    y.columns = ['target']

    Xy = X.join(y)

    return X, y, Xy, ft_grid_names, params