# Opens the default audio devices and runs a VAD and a sound classifier on them


import argparse
import numpy as np
import pyaudio
import os
import as_classification.ann_models
import as_sound.detectors.VAD_nn
import as_sound.features.extractFeatures

parser = argparse.ArgumentParser(description='Classify input speech')

parser.add_argument('vadName', help='The class name from as_sound/VAD')
parser.add_argument('vadArgs', help='The arguments to the VAD')

#parser.add_argument('classifierName', help='The class name from as_classification/models')
#parser.add_argument('classifierModel', help='The path to the model checkpoint file')

args = parser.parse_args()

CHUNK = 2048
HOP_SIZE = 0
WIDTH = 2
DTYPE = np.int16
MAX_INT = 32768.0
CHANNELS = 1
RATE = 8000

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

vad = as_sound.detectors.VAD_nn.VAD_nn(RATE)
vad.model = as_classification.ann_models.ANN_5FCL()
vad.model.initialize(15, 2)
vad.model.loadCheckpoint(os.path.dirname(os.path.realpath(__file__)) + '/data/vadModel_ANN_5FCL.chkp')


wasSilent = True

while True:
    # read audio
    string_audio_data = stream.read(CHUNK, exception_on_overflow = False)
    audio_data = np.fromstring(string_audio_data, dtype=DTYPE)
    normalized_data = audio_data / MAX_INT

    feat = as_sound.features.extractFeatures.computeSupervector(normalized_data)
    feat_rev= np.swapaxes(feat, 0, 1)[0, :]

    prob_silent = vad.model.predict([feat_rev])[0]

    isSilent = False
    if (prob_silent[0] < 0.5):
        isSilent = True

    if (wasSilent != isSilent):
        if (isSilent):
            print("Now silent")
        else:
            print("Now voiced")

        wasSilent = isSilent

    #audio_data = np.array(np.round_(synth[CHUNK:] * MAX_INT), dtype=DTYPE)
    #string_audio_data = audio_data.tostring()
    #stream.write(string_audio_data, CHUNK)

print("* done")

stream.stop_stream()
stream.close()
p.terminate()

