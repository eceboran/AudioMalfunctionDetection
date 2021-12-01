import librosa

def get_mel_spectrogram(file_path, window, overlap=None, n_fft=None, n_mels=32, no_channel=None, machine=None):
    
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
    
    # Window length in samples
    window_length = int(window*fs)
    
    # Default overlap is 50% of the window size
    if overlap==None:
        overlap = 0.5
    
    # Overlap in samples
    overlap_length = int(window_length*overlap)
    
    # Hop length in samples
    hop_length = window_length-overlap_length
    
    # Default n_fft is the smallest power of 2 larger than win_length
    if n_fft==None:
        n_fft = int(2**np.ceil(np.log2(window_length)))
   
    # Compute mel spectogram
    mel_spect = librosa.feature.melspectrogram(y=signal, sr=fs, 
                                               win_length=window_length, 
                                               hop_length=hop_length,
                                               n_fft=n_fft,
                                               n_mels=n_mels)
    # Mel spectogram in decibels
    mel_spect_db = librosa.power_to_db(mel_spect, ref=1.0, amin=sys.float_info.epsilon, top_db=np.inf)

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mels,
                                win_length=window_length, 
                                hop_length=hop_length,
                                n_fft=n_fft)
    
    params = {}
    params['window'] = window
    params['window_length'] = window_length
    params['overlap'] = overlap
    params['overlap_length'] = overlap_length
    params['hop_length'] = hop_length
    params['n_fft'] = n_fft
    params['n_mels'] = n_mels
    
    return mel_spect, mel_spect_db, mfcc, params