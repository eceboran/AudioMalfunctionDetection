import os
import numpy as np
from utils.get_mel_spectrogram import get_mel_spectrogram

def get_melspec_features_from_files(data_dir, df_input, window, n_mels, overlap=0.5, feature_type='mel_spect_db'):
    
    X = []
    y = []

    for idx, row in df_input.iterrows():
        file_path = os.path.join(data_dir, row.file_rel_path)
        fs = row.fs_Hz

        mel_spect, mel_spect_db, mfcc, params = get_mel_spectrogram(file_path, 
                                                                    window=window, 
                                                                    overlap=overlap,
                                                                    n_mels=n_mels, 
                                                                    machine=row.machine)
        
        if feature_type=='mel_spect_db':
            sample_features = mel_spect_db.flatten().reshape(-1, 1)    
        
        sample_target = row.anomaly
        X.append(sample_features)
        y.append(sample_target)
        
    X = np.array(X)
    X = X[:,:,0]
    y = np.array(y).reshape(-1, 1)
        
    return X, y, params