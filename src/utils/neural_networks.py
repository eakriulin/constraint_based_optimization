import numpy as np

def normalize_data(X, mean=None, std=None):
    mean = np.mean(X) if mean is None else mean
    std = np.std(X) if std is None else std

    return (X - mean) / std, mean, std

def layer(features_in, features_out):
    W = np.random.randn(features_in, features_out).astype(dtype=np.float64)
    b = np.random.randn(features_out).astype(dtype=np.float64)

    return (W, b)

def train(nn, X, Y, number_of_epochs, learning_rate):
    for e in range(0, number_of_epochs):
        As, Zs = forward(nn, X)
        loss = MSE(As[-1], Y)

        print(f"epoch {e + 1}, loss {loss}")

        loss_derivative = MSE(As[-1], Y, as_derivative_wrt_A=True)
        gradients, _, _ = backward(nn, X, As, Zs, loss_derivative)

        update_parameters(nn, gradients, learning_rate)

def train_with_constraint(nn_1, nn_2, nn_3, X_1_and_3, Y_1, Y_2_and_3, number_of_epochs, learning_rate):
    for e in range(0, number_of_epochs):
        As_1, Zs_1 = forward(nn_1, X_1_and_3)
        As_2, Zs_2 = forward(nn_2, As_1[-1])
        As_3, Zs_3 = forward(nn_3, X_1_and_3)

        loss_1 = MSE(As_1[-1], Y_1)
        loss_2 = MSE(As_2[-1], Y_2_and_3)
        loss_3 = MSE(As_3[-1], Y_2_and_3)
        loss_constraint = MSE(As_3[-1], As_2[-1])

        loss = loss_1 + loss_2 + loss_3 + loss_constraint
        print(f"epoch {e + 1}, loss {loss}")
        print(f"    loss 1 {loss_1}, loss 2 {loss_2}, loss 3 {loss_3}, loss constraint {loss_constraint}")

        gradients_1 = []
        gradients_2 = []
        gradients_3 = []

        # note: loss_2 wrt nn_2
        loss_2_derivative_wrt_A = MSE(As_2[-1], Y_2_and_3, as_derivative_wrt_A=True)
        loss_2_gradients_wrt_nn_2, loss_2_last_delta_wrt_nn_2, loss_2_last_W_wrt_nn_2 = backward(nn_2, As_1[-1], As_2, Zs_2, loss_derivative=loss_2_derivative_wrt_A)
        gradients_2.append(loss_2_gradients_wrt_nn_2)

        # note: loss_constraint wrt nn_2
        loss_constraint_derivative_wrt_Y = MSE(As_3[-1], As_2[-1], as_derivative_wrt_Y=True)
        loss_constraint_gradients_wrt_nn_2, loss_constraint_last_delta_wrt_nn_2, loss_constraint_last_W_wrt_nn_2 = backward(nn_2, As_1[-1], As_2, Zs_2, loss_derivative=loss_constraint_derivative_wrt_Y)
        gradients_2.append(loss_constraint_gradients_wrt_nn_2)

        # note: loss_3 wrt nn_3
        loss_3_derivative_wrt_A = MSE(As_3[-1], Y_2_and_3, as_derivative_wrt_A=True)
        loss_3_gradients_wrt_nn_3, _, _ = backward(nn_3, X_1_and_3, As_3, Zs_3, loss_derivative=loss_3_derivative_wrt_A)
        gradients_3.append(loss_3_gradients_wrt_nn_3)

        # note: loss_constraint wrt nn_3
        loss_constraint_derivative_wrt_A = MSE(As_3[-1], As_2[-1], as_derivative_wrt_A=True)
        loss_constraint_gradients_wrt_nn_3, _, _ = backward(nn_3, X_1_and_3, As_3, Zs_3, loss_derivative=loss_constraint_derivative_wrt_A)
        gradients_3.append(loss_constraint_gradients_wrt_nn_3)

        # note: loss_1 wrt nn_1
        loss_1_derivative_wrt_A = MSE(As_1[-1], Y_1, as_derivative_wrt_A=True)
        loss_1_gradients_wrt_nn_1, _, _ = backward(nn_1, X_1_and_3, As_1, Zs_1, loss_derivative=loss_1_derivative_wrt_A)
        gradients_1.append(loss_1_gradients_wrt_nn_1)

        # note: loss_2 wrt nn_1
        loss_2_gradients_wrt_nn_1, _, _ = backward(nn_1, X_1_and_3, As_1, Zs_1, delta=loss_2_last_delta_wrt_nn_2, W_of_next_layer=loss_2_last_W_wrt_nn_2)
        gradients_1.append(loss_2_gradients_wrt_nn_1)

        # note: loss_constraint wrt nn_1
        loss_constraint_gradients_wrt_nn_1, _, _ = backward(nn_1, X_1_and_3, As_1, Zs_1, delta=loss_constraint_last_delta_wrt_nn_2, W_of_next_layer=loss_constraint_last_W_wrt_nn_2)
        gradients_1.append(loss_constraint_gradients_wrt_nn_1)

        gradients_1 = reduce_gradients(gradients_1)
        gradients_2 = reduce_gradients(gradients_2)
        gradients_3 = reduce_gradients(gradients_3)

        update_parameters(nn_1, gradients_1, learning_rate)
        update_parameters(nn_2, gradients_2, learning_rate)
        update_parameters(nn_3, gradients_3, learning_rate)

def forward(nn, X):
    number_of_layers = len(nn)
    last_layer_idx = number_of_layers - 1

    Zs = [None] * number_of_layers # note: layer input x layer weights + bias
    As = [None] * number_of_layers # note: layer activations

    layer_input = X

    for l in range(0, number_of_layers):
        W, b = nn[l]
        layer_input = X if l == 0 else As[l - 1]

        Zs[l] = np.matmul(layer_input, W) + b
        As[l] = Zs[l] if l == last_layer_idx else sigmoid(Zs[l]) # note: linear activation on last later

    return As, Zs

def backward(nn, X, As, Zs, loss_derivative=None, delta=None, W_of_next_layer=None):
    assert((loss_derivative is not None) != (delta is not None))

    number_of_layers = len(nn)
    last_layer_idx = number_of_layers - 1

    gradients = [None] * number_of_layers

    for l in range(last_layer_idx, -1, -1):
        if l == last_layer_idx and loss_derivative is not None:
            delta = loss_derivative # note: linear activation on last later
        else:
            W_of_next_layer, _ = (W_of_next_layer.T, None) if W_of_next_layer is not None else nn[l + 1]
            delta = np.matmul(delta, W_of_next_layer.T) * sigmoid(Zs[l], as_derivative_wrt_Z=True)

        W_gradient = None
        if l != 0:
            W_gradient = np.matmul(As[l - 1].T, delta)
        else:
            W_gradient = np.matmul(X.T, delta)

        b_gradient = np.sum(delta, axis=0)

        gradients[l] = (W_gradient, b_gradient)

    return gradients, delta, W_of_next_layer

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

def update_parameters(nn, gradients, learning_rate):
    number_of_layers = len(nn)

    for l in range(0, number_of_layers):
        W, b = nn[l]
        W_grad, b_grad = gradients[l]

        W -= learning_rate * W_grad
        b -= learning_rate * b_grad

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

def R2(A, Y):
    sse = np.sum((A - Y) ** 2)

    mean_Y = np.mean(Y)
    sst = np.sum((Y - mean_Y) ** 2)

    return 1 - (sse / sst)

def eval(nn, X, Y):
    As, _ = forward(nn, X)
    return R2(As[-1], Y)
