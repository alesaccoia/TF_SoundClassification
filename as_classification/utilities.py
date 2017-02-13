import numpy as np

def divideTrainingData(training_data, proportion):
    INPUT_DIMS = training_data["data"].shape[1]
    INPUT_LENGTH = training_data["data"].shape[0]
    indices = np.random.permutation(INPUT_LENGTH)
    numberOfTests = int(INPUT_LENGTH * (1.0 - proportion))
    print(training_data["data"].shape)
    print(numberOfTests )
    training_idx, test_idx = indices[:-numberOfTests], indices[numberOfTests:]

    test_data = {
        "labels": training_data["labels"][training_idx,:],
        "data": training_data["data"][training_idx,:]
    }
    training_data["labels"] = training_data["labels"][test_idx,:]
    training_data["data"] = training_data["data"][test_idx,:]
    trainingDataLength = len(training_data["data"])

    return training_data, test_data
