import argparse
import src.neural_networks.height_age as height_age
import src.neural_networks.age_weight as age_weight
import src.neural_networks.height_weight as height_weight
import src.utils.neural_networks as nn_utils
import src.utils.data as data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_dataset', action='store_true')
    parser.add_argument('--height_age', action='store_true')
    parser.add_argument('--age_weight', action='store_true')
    parser.add_argument('--height_weight', action='store_true')
    parser.add_argument('--constraint', action='store_true')
    args = parser.parse_args()

    if args.generate_dataset:
        data_utils.generate_data()

    X_height_age, Y_height_age = height_age.initialize_inputs_and_targets()
    X_height_age, mean_height_age, std_height_age = data_utils.normalize_data(X_height_age)
    height_age_nn = height_age.initialize_neural_network()

    X_age_weight, Y_age_weight = age_weight.initialize_inputs_and_targets()
    X_age_weight, mean_age_weight, std_age_weight = data_utils.normalize_data(X_age_weight)
    age_weight_nn = age_weight.initialize_neural_network()

    X_height_weight, Y_height_weight = height_weight.initialize_inputs_and_targets()
    X_height_weight, mean_height_weight, std_height_weight = data_utils.normalize_data(X_height_weight)
    height_weight_nn = height_weight.initialize_neural_network()

    if args.height_age:
        nn_utils.train(height_age_nn, X_height_age, Y_height_age, number_of_epochs=10000, learning_rate=0.005)
        input, _, _ = data_utils.normalize_data([170], mean_height_age, std_height_age)
        print(nn_utils.predict(height_age_nn, [input]))

    if args.age_weight:
        nn_utils.train(age_weight_nn, X_age_weight, Y_age_weight, number_of_epochs=10000, learning_rate=0.005)
        input, _, _ = data_utils.normalize_data([1], mean_age_weight, std_age_weight)
        print(nn_utils.predict(age_weight_nn, [input]))

    if args.height_weight:
        nn_utils.train(height_weight_nn, X_height_weight, Y_height_weight, number_of_epochs=10000, learning_rate=0.005)
        input, _, _ = data_utils.normalize_data([170], mean_height_weight, std_height_weight)
        print(nn_utils.predict(height_weight_nn, [input]))

    if args.constraint:
        nn_utils.train_with_constraint(height_age_nn, age_weight_nn, height_weight_nn, X_height_age, Y_height_age, Y_height_weight, number_of_epochs=20000, learning_rate=0.001)

        input, _, _ = data_utils.normalize_data([170], mean_height_age, std_height_age)
        print(nn_utils.predict(height_age_nn, [input]))

        input, _, _ = data_utils.normalize_data([1], mean_age_weight, std_age_weight)
        print(nn_utils.predict(age_weight_nn, [input]))

        input, _, _ = data_utils.normalize_data([170], mean_height_weight, std_height_weight)
        print(nn_utils.predict(height_weight_nn, [input]))