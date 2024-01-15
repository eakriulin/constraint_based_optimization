import argparse
import src.neural_networks.height_age as height_age
import src.neural_networks.age_weight as age_weight
import src.neural_networks.height_weight as height_weight
import src.utils.neural_networks as nn_utils
import src.utils.data as data_utils

NUMBER_OF_EPOCHS = 20000
LEARNING_RATE = 0.001

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_dataset', action='store_true')
    parser.add_argument('--height_age', action='store_true')
    parser.add_argument('--age_weight', action='store_true')
    parser.add_argument('--height_weight', action='store_true')
    parser.add_argument('--constraint', action='store_true')
    args = parser.parse_args()

    if args.generate_dataset:
        data_utils.generate_dataset()

    X_height, Y_age = height_age.initialize_inputs_and_targets()
    X_age, Y_weight = age_weight.initialize_inputs_and_targets()

    X_height, mean_height, std_height = data_utils.normalize_data(X_height)
    X_age, mean_age, std_age = data_utils.normalize_data(X_age)

    nn_height_age = height_age.initialize_neural_network()
    nn_age_weight = age_weight.initialize_neural_network()
    nn_height_weight = height_weight.initialize_neural_network()

    if args.height_age:
        nn_utils.train(nn_height_age, X_height, Y_age, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.age_weight:
        nn_utils.train(nn_age_weight, X_age, Y_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.height_weight:
        nn_utils.train(nn_height_weight, X_height, Y_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.constraint:
        nn_utils.train_with_constraint(nn_height_age, nn_age_weight, nn_height_weight, X_height, Y_age, Y_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)