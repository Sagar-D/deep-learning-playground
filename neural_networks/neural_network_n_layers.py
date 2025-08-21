import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def calculate_activation(X, W, b):
    """Calculate activation vector
    X = Input Vector of shape (n, m)
    W = Weight matrix of shape (l, n)
    b = Bias matrix of shape (l, 1)
    """
    Z = np.dot(W, X) + b
    return Z, sigmoid(Z)


def forward_propogation(X, W, b):
    """Perform forward propogation.
    X = Input Vector of shape (n, m)
    W = List of size l containg Weight vector of each layer
    b = List of size l containg bias vector of each layer
    where,
        l is number of layers in the Neural network
        n is number of features in each training sample
        m is number of training samples
    """

    A = []
    Z = []
    A.append(X)
    Z.append(None)
    for i in range(1, len(W)):
        Z_current, A_current = calculate_activation(A[i - 1], W[i], b[i])
        A.append(A_current)
        Z.append(Z_current)

    return Z, A


def get_derivative_of_activation(Z, activation_method="sigmoid"):
    """Calculate the derivative of activation function.
    Supported activation functions - sigmoid
    """
    activation_method = activation_method.strip().lower()
    if activation_method == "sigmoid":
        return sigmoid(Z) * (1 - sigmoid(Z))


def calculate_cost(Y, A):
    """Calculate cost of NN w.r.t to expected output
    Y = Expected output vector of shape (1,m)
    A = Predicted output vector of shape (1,m)
    """

    epsilon = 1e-15
    A = np.clip(A, epsilon, (1 - epsilon))
    m = Y.shape[1]
    cost = -1 / m * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))
    return cost


def backward_propogation(W, Z, A, Y):
    """Calculate derivatives for all layers of Neural Network
    W = List of size l containg Weight vector of each layer
    Z = List of size l containg Z vectors of each layer
    A = List of size l containg Weight vector of each layer
    Y = Expected output vector of shape (1,m)
    """
    dZ = []
    dW = []
    db = []
    m = Y.shape[1]
    dZ.append(A[-1] - Y)
    dW.append(1 / m * np.dot(dZ[0], A[-2].T))
    db.append(1 / m * np.sum(dZ[0], axis=1, keepdims=True))
    for i in range(len(W) - 2, 0, -1):
        dZ_current = np.dot(W[i + 1].T, dZ[0]) * get_derivative_of_activation(Z[i])
        dZ.insert(0, dZ_current)
        dW.insert(0, 1 / m * np.dot(dZ_current, A[i-1].T))
        db.insert(0, 1 / m * np.sum(dZ_current, axis=1, keepdims=True))

    dW.insert(0, None)
    db.insert(0, None)

    return dW, db


def train_model(X, Y, layer_dimension, min_cost_delta=1e-7, max_learnining_iterations=100000, learning_rate=0.05):
    """Train n-layer Neural Network model.
    X = Input vector of shape (n, m)
    Y = Expected output vector of shape (1, m)
    layer_dimension = List of size l containg number of nodes in each layer
    where,
        l is number of layers in the Neural network
        n is number of features in each training sample
        m is number of training samples
    """

    W = []
    b = []
    W.append(None)
    W.append(np.random.rand(layer_dimension[0], X.shape[0]))
    b.append(None)
    b.append(np.zeros((layer_dimension[0], 1)))
    for i in range(1, len(layer_dimension)):
        W.append(np.random.rand(layer_dimension[i], layer_dimension[i - 1]))
        b.append(np.zeros((layer_dimension[i], 1)))

    cost = 0.5
    prev_cost = float("inf")

    for learning_iteration in range(0, max_learnining_iterations):
        
        Z, A = forward_propogation(X, W, b)
        cost = calculate_cost(Y,A[-1])
        dW, db = backward_propogation(W, Z, A, Y)
        
        if abs(prev_cost-cost) <= min_cost_delta :
            print(f"Training stopped at iteration {learning_iteration}. No significant change in cost across learning iterations")
            break
        
        is_gradient_decent_achived = True
        for i in range(1, len(dW)) :
            if np.linalg.norm(dW[i]) >= 1e-6 or np.linalg.norm(db[i]) >= 1e-6 :
                is_gradient_decent_achived = False
        if is_gradient_decent_achived :
            print(f"Training stopped at iteration {learning_iteration}. Gradient descent complete.")
            break
        
        for i in range(1, len(W)) :
            W[i] = W[i] - learning_rate * dW[i]
            b[i] = b[i] - learning_rate * db[i]
        prev_cost = cost

        if learning_iteration % 10 == 0 :
            print(f"Itertion {learning_iteration} completed. Cost = {cost}")
    
    print("Model Training completed.......!!!!")
    print("--"*50)
    print(f"Cost : {cost}")
    print("--"*50)
    
    for i in range(1, len(W)) :
        print(f"\n----- Layer {i} ----- \nWeights : {W[i]} \nBias : {b[i]}")
    
    return W, b

def predict(X, W, b) :
    Z, A = forward_propogation(X, W, b)
    return A[-1]

if __name__ == "__main__" :
    x = np.random.rand(3, 10)
    y = np.random.rand(1, 10)
    lcount = [8,5,3,1]
    train_model(x, y, lcount)
