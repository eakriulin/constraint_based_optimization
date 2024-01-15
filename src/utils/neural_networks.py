import numpy as np

def layer(features_in, features_out):
    W = np.random.randn(features_in, features_out).astype(dtype=np.float64)
    b = np.random.randn(features_out).astype(dtype=np.float64)

    return (W, b)

def train(neural_network, X, Y, number_of_epochs, learning_rate):
    for e in range(0, number_of_epochs):
        As, Zs = forward(neural_network, X)
        loss = MSE(As[-1], Y)

        print(f"epoch {e + 1}, loss {loss}")

        loss_derivative = MSE(As[-1], Y, as_derivative_wrt_A=True)
        gradients, _, _ = backward(neural_network, X, As, Zs, loss_derivative)

        update_parameters(neural_network, gradients, learning_rate)

def train_with_constraint(
    nn_height_age,
    nn_age_weight,
    nn_height_weight,
    X_height,
    Y_age,
    Y_weight,
    number_of_epochs,
    learning_rate,
):
    for e in range(0, number_of_epochs):
        As_height_age, Zs_height_age = forward(nn_height_age, X_height)
        As_age_weight, Zs_age_weight = forward(nn_age_weight, As_height_age[-1])
        As_height_weight, Zs_height_weight = forward(nn_height_weight, X_height)

        loss_height_age = MSE(As_height_age[-1], Y_age)
        loss_age_weight = MSE(As_age_weight[-1], Y_weight)
        loss_height_weight = MSE(As_height_weight[-1], Y_weight)
        loss_constraint = MSE(As_height_weight[-1], As_age_weight[-1])

        loss = loss_height_age + loss_age_weight + loss_height_weight + loss_constraint
        print(f"epoch {e + 1}, loss {loss}")
        print(f"\tloss_height_age {loss_height_age}, loss_age_weight {loss_age_weight}, loss_height_weight {loss_height_weight}, loss_constraint {loss_constraint}")

        W1_gradients = []
        W2_gradients = []
        W3_gradients = []

        # loss_age_weight wrt W2
        loss_age_weight_derivative_wrt_A = MSE(As_age_weight[-1], Y_weight, as_derivative_wrt_A=True)
        W2_gradients_for_age_weight, W2_delta_for_age_weight, W2_W_for_age_weight = backward(nn_age_weight, As_height_age[-1], As_age_weight, Zs_age_weight, loss_derivative=loss_age_weight_derivative_wrt_A)
        W2_gradients.append(W2_gradients_for_age_weight)

        # loss_constraint wrt W2
        loss_constraint_derivative_wrt_Y = MSE(As_height_weight[-1], As_age_weight[-1], as_derivative_wrt_Y=True)
        W2_gradients_for_constraint, W2_delta_for_constraint, W2_W_for_constraint = backward(nn_age_weight, As_height_age[-1], As_age_weight, Zs_age_weight, loss_derivative=loss_constraint_derivative_wrt_Y)
        W2_gradients.append(W2_gradients_for_constraint)

        # loss_height_weight wrt W3
        loss_height_weight_derivative_wrt_A = MSE(As_height_weight[-1], Y_weight, as_derivative_wrt_A=True)
        W3_gradients_for_height_weight, _, _ = backward(nn_height_weight, X_height, As_height_weight, Zs_height_weight, loss_derivative=loss_height_weight_derivative_wrt_A)
        W3_gradients.append(W3_gradients_for_height_weight)

        # loss_constraint wrt W3
        loss_constraint_derivative_wrt_A = MSE(As_height_weight[-1], As_age_weight[-1], as_derivative_wrt_A=True)
        W3_gradients_for_constraint, _, _ = backward(nn_height_weight, X_height, As_height_weight, Zs_height_weight, loss_derivative=loss_constraint_derivative_wrt_A)
        W3_gradients.append(W3_gradients_for_constraint)

        # loss_height_age wrt W1
        loss_height_age_derivative_wrt_A = MSE(As_height_age[-1], Y_age, as_derivative_wrt_A=True)
        W1_gradients_for_height_age, _, _ = backward(nn_height_age, X_height, As_height_age, Zs_height_age, loss_derivative=loss_height_age_derivative_wrt_A)
        W1_gradients.append(W1_gradients_for_height_age)

        # loss_age_weight wrt W1
        W1_gradients_for_age_weight, _, _ = backward(nn_height_age, X_height, As_height_age, Zs_height_age, delta=W2_delta_for_age_weight, W_of_succeeding_layer=W2_W_for_age_weight)
        W1_gradients.append(W1_gradients_for_age_weight)

        # loss_constraint wrt W1
        W1_gradients_for_constraint, _, _ = backward(nn_height_age, X_height, As_height_age, Zs_height_age, delta=W2_delta_for_constraint, W_of_succeeding_layer=W2_W_for_constraint)
        W1_gradients.append(W1_gradients_for_constraint)

        W1_reduced_gradients = reduce_gradients(W1_gradients)
        W2_reduced_gradients = reduce_gradients(W2_gradients)
        W3_reduced_gradients = reduce_gradients(W3_gradients)

        update_parameters(nn_height_age, W1_reduced_gradients, learning_rate)
        update_parameters(nn_age_weight, W2_reduced_gradients, learning_rate)
        update_parameters(nn_height_weight, W3_reduced_gradients, learning_rate)

def forward(neural_network, X):
    number_of_layers = len(neural_network)
    last_layer_idx = number_of_layers - 1

    Zs = [None] * number_of_layers # note: layer input x layer weights + bias
    As = [None] * number_of_layers # note: layer activations

    layer_input = X

    for l in range(0, number_of_layers):
        W, b = neural_network[l]
        layer_input = X if l == 0 else As[l - 1]

        Zs[l] = np.matmul(layer_input, W) + b
        As[l] = Zs[l] if l == last_layer_idx else sigmoid(Zs[l]) # note: linear activation on last later

    return As, Zs

def backward(neural_network, X, As, Zs, loss_derivative=None, delta=None, W_of_succeeding_layer=None):
    number_of_layers = len(neural_network)
    last_layer_idx = number_of_layers - 1

    gradients = [None] * number_of_layers

    for l in range(last_layer_idx, -1, -1):
        if l == last_layer_idx and loss_derivative is not None:
            delta = loss_derivative # note: linear activation on last later
        else:
            W_of_succeeding_layer, _ = (W_of_succeeding_layer.T, None) if W_of_succeeding_layer is not None else neural_network[l + 1]
            delta = np.matmul(delta, W_of_succeeding_layer.T) * sigmoid(Zs[l], as_derivative_wrt_Z=True)

        W_gradient = None
        if l != 0:
            W_gradient = np.matmul(As[l - 1].T, delta)
        else:
            W_gradient = np.matmul(X.T, delta)

        b_gradient = np.sum(delta, axis=0)

        gradients[l] = (W_gradient, b_gradient)

    return gradients, delta, W_of_succeeding_layer

def update_parameters(neural_network, gradients, learning_rate):
    number_of_layers = len(neural_network)

    for l in range(0, number_of_layers):
        W, b = neural_network[l]
        W_grad, b_grad = gradients[l]

        W -= learning_rate * W_grad
        b -= learning_rate * b_grad

def reduce_gradients(all_gradients):
    number_of_layers = len(all_gradients[0])
    reduced_gradients = [None] * number_of_layers

    for gradients in all_gradients:
        for l in range(0, number_of_layers):
            W_grad, b_grad = gradients[l]

            if reduced_gradients[l] is None:
                reduced_gradients[l] = [W_grad, b_grad]
            else:
                reduced_gradients[l][0] += W_grad
                reduced_gradients[l][1] += b_grad

    return reduced_gradients


def sigmoid(Z, as_derivative_wrt_Z=False):
    if (as_derivative_wrt_Z):
        return np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)
    
    return 1 / (1 + np.exp(-Z))

def MSE(A, Y, as_derivative_wrt_A=False, as_derivative_wrt_Y=False):
    if as_derivative_wrt_A:
        return (A - Y) * 2 / len(Y)

    if as_derivative_wrt_Y:
        return (A - Y) * -2 / len(Y)

    return np.mean((A - Y) ** 2)

def predict(neural_network, input):
    As, _ = forward(neural_network, input)
    return As[-1]