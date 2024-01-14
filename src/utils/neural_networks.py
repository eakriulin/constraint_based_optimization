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

        loss_derivative = MSE(As[-1], Y, as_derivative=True)
        gradients = backward(neural_network, X, As, Zs, loss_derivative)

        update_parameters(neural_network, gradients, learning_rate)

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

def backward(neural_network, X, As, Zs, loss_derivative):
    number_of_layers = len(neural_network)
    last_layer_idx = number_of_layers - 1

    gradients = [None] * number_of_layers
    delta = None

    for l in range(last_layer_idx, -1, -1):
        if l == last_layer_idx:
            delta = loss_derivative # note: linear activation on last later
        else:
            W_of_succeeding_layer, _ = neural_network[l + 1]
            delta = np.matmul(delta, W_of_succeeding_layer.T) * sigmoid(Zs[l], as_derivative=True)

        W_gradient = None
        if l != 0:
            W_gradient = np.matmul(As[l - 1].T, delta)
        else:
            W_gradient = np.matmul(X.T, delta)

        b_gradient = np.sum(delta, axis=0)

        gradients[l] = (W_gradient, b_gradient)

    return gradients

def update_parameters(neural_network,gradients, learning_rate):
    number_of_layers = len(neural_network)

    for l in range(0, number_of_layers):
        W, b = neural_network[l]
        W_grad, b_grad = gradients[l]

        W -= learning_rate * W_grad
        b -= learning_rate * b_grad

def sigmoid(Z, as_derivative=False):
    if (as_derivative):
        return np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)
    
    return 1 / (1 + np.exp(-Z))

def MSE(A, Y, as_derivative=False):
    if as_derivative:
        return (A - Y) * 2 / len(Y)

    return np.mean((A - Y) ** 2)

def predict(neural_network, input):
    As, _ = forward(neural_network, input)
    return As[-1]