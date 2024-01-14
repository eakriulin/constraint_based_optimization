import os
import numpy as np
import src.utils.neural_networks as nn_utils

def initialize_inputs_and_targets():
    data = np.genfromtxt(os.path.join('dataset', 'age_height_weight_dataset.csv'), delimiter=',', dtype=np.float64)

    height = data[1:, 1]
    weight = data[1:, 2]

    X = height.reshape(len(height), 1).astype(dtype=np.float64)
    Y = weight.reshape(len(weight), 1).astype(dtype=np.float64)

    return X, Y

def initialize_neural_network():
    return [
        nn_utils.layer(1, 5),
        nn_utils.layer(5, 1),
    ]