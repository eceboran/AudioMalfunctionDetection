import numpy as np
import librosa

def get_mel_frequency_bands(fs, n_mels, n_fft = 1000, fmin=0, fmax=None):

    # Default min frequency is zero
    # Default max frequency is half of the sampling frequency
    if fmax == None:
        fmax = fs / 2

    # Default n_fft is 1000

    # Centers of mel filter bands
    filter_banks = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)
    freq_axis = np.linspace(fmin, fmax, n_fft // 2 + 1)
    mel_center_freq = freq_axis[np.argmax(filter_banks, axis=1)]
    
    
    return mel_center_freq, freq_axis