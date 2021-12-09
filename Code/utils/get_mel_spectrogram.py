import sys
import numpy as np
import numpy.matlib
import librosa
from utils.get_mel_frequency_bands import get_mel_frequency_bands


def get_mel_spectrogram(file_path, window, overlap=0.5, n_fft=None, n_mels=32, fmin=0, fmax=None, no_channel=None, machine=None, normalize_signal=False):
    
    # Load the signal
    signal, fs = librosa.load(file_path, sr=None, mono=False)
    
    # Default channel no selected based on machine
    channel_dict = {'pump': 3, 'valve': 1, 'slider': 7, 'fan': 5}
    if no_channel == None:
        if machine == None:
            signal = signal.mean(axis=0)
        else:
            no_channel = channel_dict[machine]-1
            signal = signal[no_channel, :]
    else:
        signal = signal[no_channel, :]
    
    # Normalize the signal
    if normalize_signal == 'z-score':
        signal = (signal - signal.mean()) / signal.std()

    # Window length in samples
    window_length = int(window*fs)
    
    # Overlap in samples
    overlap_length = int(window_length*overlap)
    
    # Hop length in samples
    hop_length = window_length-overlap_length
    
    # Default n_fft is the smallest power of 2 larger than win_length
    if n_fft==None:
        n_fft = int(2**np.ceil(np.log2(window_length)))
    
    # Default min frequency is zero
    # Default max frequency is half of the sampling frequency
    if fmax==None:
        fmax = fs/2
    
    # Compute mel spectogram
    mel_spect = librosa.feature.melspectrogram(y=signal, sr=fs, 
                                               win_length=window_length, 
                                               hop_length=hop_length,
                                               n_fft=n_fft,
                                               n_mels=n_mels,
                                               fmin=fmin,
                                               fmax=fmax)
    # Mel spectogram in decibels
    mel_spect_db = librosa.power_to_db(mel_spect, ref=1.0, amin=sys.float_info.epsilon, top_db=np.inf)

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mels,
                                win_length=window_length, 
                                hop_length=hop_length,
                                n_fft=n_fft,
                                fmin=fmin,
                                fmax=fmax)

    # Centers of time windows TODO
    no_windows = mel_spect.shape[1]

    # Centers of mel filter bands
    mel_center_freq, freq_axis = get_mel_frequency_bands(fs, n_mels, n_fft=n_fft, fmin=fmin, fmax=fmax)

    # Frequency and time indices of point in the grid
    no_window_grid = np.matlib.repmat(np.arange(0, no_windows).reshape(1, -1), n_mels, 1)
    no_freq_grid = np.matlib.repmat(np.arange(0, n_mels).reshape(-1, 1), 1, no_windows)
    ft_grid_names = np.empty(no_window_grid.shape).astype(str)
    for i in range(0, no_window_grid.shape[0]):
        for j in range(0, no_window_grid.shape[1]):
            ft_grid_names[i, j] = f"{no_freq_grid[i, j]}_{no_window_grid[i, j]}"

    params = {}
    params['window'] = window
    params['window_length'] = window_length
    params['no_windows'] = no_windows
    params['overlap'] = overlap
    params['overlap_length'] = overlap_length
    params['hop_length'] = hop_length
    params['n_fft'] = n_fft
    params['n_mels'] = n_mels
    params['fs'] = fs
    params['fmin'] = fmin
    params['fmax'] = fmax
    params['mel_center_freq'] = mel_center_freq
    params['ft_grid_names'] = ft_grid_names
    
    return mel_spect, mel_spect_db, mfcc, params