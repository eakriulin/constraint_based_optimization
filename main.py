import argparse
import src.neural_networks.height_age as height_age
import src.neural_networks.age_weight as age_weight
import src.neural_networks.height_weight as height_weight
import src.utils.neural_networks as nn_utils
import src.utils.dataset as dataset_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_dataset', action='store_true', help='If passed, the dataset will be generated')
    parser.add_argument('--height_age', action='store_true', help='If passed, the height_age nn will be trained and evaluated')
    parser.add_argument('--age_weight', action='store_true', help='If passed, the age_weight nn will be trained and evaluated')
    parser.add_argument('--height_weight', action='store_true', help='If passed, the height_weight nn will be trained and evaluated')
    parser.add_argument('--independent', action='store_true', help='If passed, all three nns will be trained independently and evaluated')
    parser.add_argument('--constraint', action='store_true', help='If passed, all three nns will be trained in constraint mode and evaluated')
    args = parser.parse_args()

    if args.generate_dataset:
        dataset_utils.generate_dataset()

    inputs_train_height, Y_train_age, inputs_test_height, Y_test_age = height_age.initialize_inputs_and_targets()
    inputs_train_age, Y_train_weight, inputs_test_age, Y_test_weight = age_weight.initialize_inputs_and_targets()

    X_train_height, mean_height, std_height = nn_utils.normalize_data(inputs_train_height)
    X_test_height, _, _ = nn_utils.normalize_data(inputs_test_height, mean_height, std_height)
    X_train_age, mean_age, std_age = nn_utils.normalize_data(inputs_train_age)
    X_test_age, _, _ = nn_utils.normalize_data(inputs_test_age, mean_age, std_age)

    nn_height_age = height_age.initialize_neural_network()
    nn_age_weight = age_weight.initialize_neural_network()
    nn_height_weight = height_weight.initialize_neural_network()

    if args.height_age:
        nn_utils.train(nn_height_age, X_train_height, Y_train_age, number_of_epochs=10000, learning_rate=0.001)
        print("\nEvaluation:")
        print(f"height->age nn: MSE = {nn_utils.eval(nn_height_age, X_test_height, Y_test_age)}")

        print("\nPrediction:")
        print(f"height->age nn: height {inputs_test_height[49:50][0][0]} cm -> age {nn_utils.predict(nn_height_age, X_test_height[49:50])[0][0]} years")

    if args.age_weight:
        nn_utils.train(nn_age_weight, X_train_age, Y_train_weight, number_of_epochs=20000, learning_rate=0.005)
        print("\nEvaluation:")
        print(f"age->weight nn: MSE = {nn_utils.eval(nn_age_weight, X_test_age, Y_test_weight)}")

        print("\nPrediction:")
        print(f"age->weight nn: age {inputs_test_age[49:50][0][0]} years -> weight {nn_utils.predict(nn_age_weight, X_test_age[49:50])[0][0]} kg")

    if args.height_weight:
        nn_utils.train(nn_height_weight, X_train_height, Y_train_weight, number_of_epochs=20000, learning_rate=0.005)
        print("\nEvaluation:")
        print(f"height->weight nn: MSE = {nn_utils.eval(nn_height_weight, X_test_height, Y_test_weight)}")

        print("\nPrediction:")
        print(f"height->weight nn: height {inputs_test_height[49:50][0][0]} cm -> weight {nn_utils.predict(nn_height_weight, X_test_height[49:50])[0][0]} kg")

    if args.independent:
        nn_utils.train(nn_height_age, X_train_height, Y_train_age, number_of_epochs=10000, learning_rate=0.001)
        nn_utils.train(nn_age_weight, nn_utils.predict(nn_height_age, X_train_height), Y_train_weight, number_of_epochs=20000, learning_rate=0.005)
        nn_utils.train(nn_height_weight, X_train_height, Y_train_weight, number_of_epochs=20000, learning_rate=0.005)
        print("\nEvaluation:")
        print(f"height->age->weight nn: MSE = {nn_utils.eval_chain(nn_height_age, nn_age_weight, X_test_height, Y_test_weight)}")
        print(f"height->age nn: MSE = {nn_utils.eval(nn_height_age, X_test_height, Y_test_age)}")
        print(f"age->weight nn: MSE = {nn_utils.eval(nn_age_weight, nn_utils.predict(nn_height_age, X_test_height), Y_test_weight)}")
        print(f"height->weight nn: MSE = {nn_utils.eval(nn_height_weight, X_test_height, Y_test_weight)}")

        print("\nPrediction:")
        print(f"height->age->weight nn: height {inputs_test_height[49:50][0][0]} cm -> weight {nn_utils.predict_chain(nn_height_age, nn_age_weight, X_test_height[49:50])[0][0]} kg")
        print(f"height->age nn: height {inputs_test_height[49:50][0][0]} cm -> age {nn_utils.predict(nn_height_age, X_test_height[49:50])[0][0]} years")
        print(f"age->weight nn: age {inputs_test_age[49:50][0][0]} years -> weight {nn_utils.predict(nn_age_weight, nn_utils.predict(nn_height_age, X_test_height[49:50]))[0][0]} kg")
        print(f"height->weight nn: height {inputs_test_height[49:50][0][0]} cm -> weight {nn_utils.predict(nn_height_weight, X_test_height[49:50])[0][0]} kg")

    if args.constraint:
        nn_utils.train_with_constraint(nn_height_age, nn_age_weight, nn_height_weight, X_train_height, Y_train_age, Y_train_weight, number_of_epochs=100000, learning_rate=0.001)
        print("\nEvaluation:")
        print(f"height->age->weight nn: MSE = {nn_utils.eval_chain(nn_height_age, nn_age_weight, X_test_height, Y_test_weight)}")
        print(f"height->age nn: MSE = {nn_utils.eval(nn_height_age, X_test_height, Y_test_age)}")
        print(f"age->weight nn: MSE = {nn_utils.eval(nn_age_weight, nn_utils.predict(nn_height_age, X_test_height), Y_test_weight)}")
        print(f"height->weight nn: MSE = {nn_utils.eval(nn_height_weight, X_test_height, Y_test_weight)}")

        print("\nPrediction:")
        print(f"height->age->weight nn: height {inputs_test_height[49:50][0][0]} cm -> weight {nn_utils.predict_chain(nn_height_age, nn_age_weight, X_test_height[49:50])[0][0]} kg")
        print(f"height->age nn: height {inputs_test_height[49:50][0][0]} cm -> age {nn_utils.predict(nn_height_age, X_test_height[49:50])[0][0]} years")
        print(f"age->weight nn: age {inputs_test_age[49:50][0][0]} years -> weight {nn_utils.predict(nn_age_weight, nn_utils.predict(nn_height_age, X_test_height[49:50]))[0][0]} kg")
        print(f"height->weight nn: height {inputs_test_height[49:50][0][0]} cm -> weight {nn_utils.predict(nn_height_weight, X_test_height[49:50])[0][0]} kg")
