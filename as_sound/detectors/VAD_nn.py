import os
import errno
import tensorflow as tf
import numpy as np
import pickle
import signal
import librosa
import soundfile
import argparse



class VAD_nn:
    def __init__(self, samplingRate):
        self.samplingRate = samplingRate
        self.model = None
    def predict(self, featuresVector):
        self.model.predict(featuresVector)

