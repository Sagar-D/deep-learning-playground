import numpy as np

SUPPORTED_ACTIVATION_METHODS = {"sigmoid"}


def sigmoid(Z):
    """Calculate the sigmoid function value for a given Vector"""
    return 1 / (1 + np.exp(-Z))


def get_derivative_of_activation(A, activation_method="sigmoid"):
    """Calculate the derivative of activation function.
        Supported activation functions - sigmoid
    """
    activation_method = activation_method.strip().lower()
    if activation_method == "sigmoid":
        return A * (1 - A)


def calculate_activation(X, W, b, activation_method="sigmoid"):
    """Calculate Activation Vector for a given Input vector, Weight and bias.
    X = Input vector of shape (n,m)
    W = Weight vector of shape (l,n)
    b = Bias Vector of shape (l,1)
    activation_method = Activation method to be used
    where,
        m = Number of training samples
        n = Number of features in one training sample
        l = Number of nodes in current layer
    """

    Z = np.dot(W, X) + b

    activation_method = activation_method.strip().lower()
    if activation_method == "sigmoid":
        return sigmoid(Z)

    raise ValueError(f"Activation method {activation_method} is not supported.")


def forward_propagation(X, W1, b1, W2, b2, activation_method="sigmoid"):
    """Perform forward propgation to predict the Yhat/A
    X = Input feature vector of shape (n,m)
    W1 = Layer-1 Weight vector of shape (l,n)
    b1 = Layer-1 Bias Vector of shape (l,1)
    W2 = Layer-2 Weight vector of shape (1,n)
    b2 = Layer-2 Bias Vector of shape (1,1)
    activation_method = Activation method for layer 1
    where,
        m = Number of training samples
        n = Number of features in one training sample
        l = Number of nodes in hidden layer (layer-1)
    """
    A1 = calculate_activation(X, W1, b1, activation_method)
    A2 = calculate_activation(A1, W2, b2)
    return A1, A2


def calculate_cost(Y, A):
    """Calculate the cost functtion value for given
    Y = Expected output vector of shape (1,m)
    A = Predicted output vector of shape (1,m)
    """
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)

    m = Y.shape[1]
    cost = -1 / m * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))
    return cost


def backward_propogation(X, Y, W2, A1, A2, a1_method="sigmoid"):
    """Calculate the derivatives of Cost function using backward propogation
    X = Input feature vector of shape (n,m)
    Y = Expected output vector of shape (1,m)
    W2 = Layer-2 Weight vector of shape (1,l)
    A1 = Output of Layer 1 of shape (l,m)
    A2 = Output of Layer 2 of shape (1,m)
    a1_method = Activation method used in layer 1
    where,
        m = Number of training samples
        n = Number of features in one training sample
        l = Number of nodes in hidden layer (layer-1)
    """

    dZ2 = A2 - Y
    dZ1 = np.dot(W2.T, dZ2) * get_derivative_of_activation(A1, a1_method)

    m = X.shape[1]
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def train_model(X, Y, l2_node_count=4, a1_method="sigmoid"):
    """Train a Binary classifier using 2-Layer Neural Network for given
    X = Input feature vector of shape (n,m)
    Y = Expected output vector of shape (1,m)
    l2_node_count = number of layer 1 nodes
    a1_method = Activation method to be used in layer 1
    where,
        m = Number of training samples
        n = Number of features in one training sample
    """

    m = X.shape[1]
    n = X.shape[0]
    l = l2_node_count
    MAX_LEARNING_ITERATION_COUNT = 100000
    cost = 0
    prev_cost = float("inf")
    alpha = 0.1

    W1 = np.random.rand(l, n) * 0.01
    b1 = np.zeros((l, 1))
    W2 = np.random.rand(1, l) * 0.01
    b2 = np.zeros((1, 1))

    for i in range(0, MAX_LEARNING_ITERATION_COUNT):

        A1, A2 = forward_propagation(X, W1, b1, W2, b2, activation_method=a1_method)
        cost = calculate_cost(Y, A2)
        dW1, db1, dW2, db2 = backward_propogation(X, Y, W2, A1, A2, a1_method=a1_method)

        if abs(prev_cost - cost) < 1e-7:
            print(
                "Stopping at iteration {i}. No significant difference in cost over iterations"
            )
            break

        if (
            np.linalg.norm(dW1) < 1e-6
            and np.linalg.norm(db1) < 1e-6
            and np.linalg.norm(dW2) < 1e-6
            and np.linalg.norm(db2) < 1e-6
        ):
            print("Stopping at iteration {i}. Gradients have flattened")
            break

        W1 = W1 - (alpha * dW1)
        b1 = b1 - (alpha * db1)
        W2 = W2 - (alpha * dW2)
        b2 = b2 - (alpha * db2)
        prev_cost = cost

        if i % 1000 == 0:
            print(f"Iteration : {i} , cost : {cost}")

    print("Model Training completed...")
    print(f"W1 = {W1} \nb1 = {b1} \nW2 = {W2} \nb2 = {b2}")

    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2, activation_method="sigmoid"):
    """Perform forward propgation to predict the Yhat/A
    X = Input feature vector of shape (n,m)
    W1 = Layer-1 Weight vector of shape (l,n)
    b1 = Layer-1 Bias Vector of shape (l,1)
    W2 = Layer-2 Weight vector of shape (1,n)
    b2 = Layer-2 Bias Vector of shape (1,1)
    activation_method = Activation method for layer 1
    where,
        m = Number of training samples
        n = Number of features in one training sample
        l = Number of nodes in hidden layer (layer-1)
    """
    A1 = calculate_activation(X, W1, b1, activation_method)
    A2 = calculate_activation(A1, W2, b2)
    return A2