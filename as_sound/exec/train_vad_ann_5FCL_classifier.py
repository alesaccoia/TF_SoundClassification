import numpy as np
import as_classification.ann_models
import as_sound.features.extractFeatures as ef
import as_classification.utilities
import os
import matplotlib.pyplot as plt


# -------------------------------
# CREATE MODEL
# -------------------------------

model = as_classification.ann_models.ANN_5FCL()
model.initialize(15,2)

# -------------------------------
# READ AUDIO FILES
# -------------------------------

speech_data = ef.computeSupervectorForFile(os.path.dirname(os.path.realpath(__file__)) + '/data/Speech.wav', 8000, 2048, 2049)
noise_data = ef.computeSupervectorForFile(os.path.dirname(os.path.realpath(__file__)) + '/data/Noise.wav', 8000, 2048, 2049)

whole_data = np.hstack((speech_data, noise_data))
whole_data = np.swapaxes(whole_data,0,1)
whole_labels = np.zeros((whole_data.shape[0], 2))
whole_labels[:speech_data.shape[1],0] = 1
whole_labels[speech_data.shape[1]:,1] = 1

training_data = {"labels": whole_labels,
                 "data": whole_data}


training_data, test_data = as_classification.utilities.divideTrainingData(training_data, 0.6)


# -------------------------------
# TRAIN
# -------------------------------

model.train(training_data, test_data, 10, 20, 100)

model.saveCheckpoint(os.path.dirname(os.path.realpath(__file__)) + '/data/vadModel_ANN_5FCL.chkp')






#xp = np.arange(0,prediction.shape[0])

#plt.plot(xp, test_data[:,14], '-b', label='RMS')
#plt.plot(xp, prediction[:,0], '-r', label='ANN Output')
#plt.legend(loc='upper left')

#plt.show()

#feat = as_sound.features.extractFeatures.computeSupervector(normalized_data)


