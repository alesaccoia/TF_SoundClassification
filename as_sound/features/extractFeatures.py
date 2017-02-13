import librosa
import soundfile
import numpy as np

def computeSupervectorForFile(filename, DEST_SR = 8000, WINDOW_SIZE = 2048, HOP_SIZE = 2048):
    predictFile = filename
    data_original, sr_original = soundfile.read(predictFile)
    if len(data_original.shape) > 1 and (data_original.shape[0] > data_original.shape[1]):
        data_original = np.swapaxes(data_original,0,1)
    data = librosa.core.resample(data_original, sr_original, DEST_SR)
    sr = DEST_SR
    sound_data = librosa.util.normalize(librosa.to_mono(data))
    return computeSupervector(sound_data, sr, WINDOW_SIZE, HOP_SIZE)

def computeSupervector(sound_data, sr = 8000, WINDOW_SIZE = 2048, HOP_SIZE = 2048):
    features_mfcc = librosa.feature.mfcc(sound_data, sr, n_mfcc=13, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE+1)
    features_zcr = librosa.feature.zero_crossing_rate(sound_data, frame_length=WINDOW_SIZE, hop_length=HOP_SIZE+1)
    features_zcr = features_zcr[0]
    features_rmse = librosa.feature.rmse(sound_data, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE+1)
    features_rmse = features_rmse[0]
    return np.vstack((features_mfcc, features_zcr, features_rmse))