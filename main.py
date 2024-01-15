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

    X_train_height, Y_train_age, X_test_height, Y_test_age = height_age.initialize_inputs_and_targets()
    X_train_age, Y_train_weight, X_test_age, Y_test_weight = age_weight.initialize_inputs_and_targets()

    X_train_height, mean_train_height, std_train_height = data_utils.normalize_data(X_train_height)
    X_test_height, mean_test_height, std_test_height = data_utils.normalize_data(X_test_height)
    X_train_age, mean_train_age, std_train_age = data_utils.normalize_data(X_train_age)
    X_test_age, mean_test_age, std_test_age = data_utils.normalize_data(X_test_age)

    nn_height_age = height_age.initialize_neural_network()
    nn_age_weight = age_weight.initialize_neural_network()
    nn_height_weight = height_weight.initialize_neural_network()

    if args.height_age:
        nn_utils.train(nn_height_age, X_train_height, Y_train_age, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.age_weight:
        nn_utils.train(nn_age_weight, X_train_age, Y_train_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.height_weight:
        nn_utils.train(nn_height_weight, X_train_height, Y_train_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)

    if args.constraint:
        nn_utils.train_with_constraint(nn_height_age, nn_age_weight, nn_height_weight, X_train_height, Y_train_age, Y_train_weight, NUMBER_OF_EPOCHS, LEARNING_RATE)