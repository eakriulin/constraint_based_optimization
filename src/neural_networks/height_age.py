import os
import numpy as np
import src.utils.neural_networks as nn_utils

def initialize_inputs_and_targets():
    data_train = np.genfromtxt(os.path.join('dataset', 'train', 'age_height_weight_dataset.csv'), delimiter=',', dtype=np.float64)
    data_test = np.genfromtxt(os.path.join('dataset', 'test', 'age_height_weight_dataset.csv'), delimiter=',', dtype=np.float64)

    height_train = data_train[1:, 1]
    age_train = data_train[1:, 0]
    height_test = data_test[1:, 1]
    age_test = data_test[1:, 0]

    X_train = height_train.reshape(len(height_train), 1).astype(dtype=np.float64)
    Y_train = age_train.reshape(len(age_train), 1).astype(dtype=np.float64)
    X_test = height_test.reshape(len(height_test), 1).astype(dtype=np.float64)
    Y_test = age_test.reshape(len(age_test), 1).astype(dtype=np.float64)

    return X_train, Y_train, X_test, Y_test

def initialize_neural_network():
    return [
        nn_utils.layer(1, 5),
        nn_utils.layer(5, 1),
    ]