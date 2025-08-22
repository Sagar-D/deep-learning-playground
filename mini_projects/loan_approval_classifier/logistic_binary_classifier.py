# Binary Classifier implemented using Logistic Regression

import numpy as np
from random import random

TRAINING_CUT_OFF_LIMIT = 10000

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def predict(X, W, b):
    """Predict the output vector A using Logistic Regression
    X = Training Data Vector of shape (m,n)
    W = Weight vector of shape (n,1)
    b = Bias
        where,
            m = Training Sample size
            n = Feature size
    """

    if X.shape[1] == None or X.shape[1] == 0 :
        ValueError("Feature dimension (n) is 0")
    if W.shape[1] == None or W.shape[1] != 1 or W.shape[0] != X.shape[1]:
        ValueError("Invalid weight matrix W")

    A = np.dot(X, W) + b
    A = sigmoid(A)
    return A


def calculate_cost(Y, A):
    """Calculate the cost value.
    Y = Expected output vector of shape (m,1)
    A = Predicted output vector of shape (m,1)
        where,
            m = Training Sample size
            n = Feature size
    """

    if Y.shape[0] != A.shape[0] :
        ValueError("Size of Matrix A and Y are different")

    def handle_ln_zero(Z):
        epsilon = 1e-15
        return np.maximum(np.minimum(Z, 1 - epsilon), epsilon)

    L = -1 * ((Y * np.log(handle_ln_zero(A))) + (1 - Y) * np.log(handle_ln_zero(1 - A)))
    return np.sum(L) / len(Y)


def calculate_cost_derivative(X, Y, A):
    """Calculate the partial derivatives of Cost function.
    X = Training Data Vector of shape (m,n)
    Y = Expected output vector of shape (m,1)
    A = Predicted output vector of shape (m,1)
        where,
            m = Training Sample size
            n = Feature size
    """

    if X.shape[1] == None or X.shape[1] == 0 :
        ValueError("Feature dimension (n) is 0")
    if Y.shape[1] == None or Y.shape[1] != 1 :
        ValueError("Invalid output label matrix Y. (Y.shape) = (m,1)")
    if X.shape[0] != Y.shape[0] != A.shape[0] :
        ValueError("Size (m) of matrix X, Y and A doesn't match")

    dW = np.dot(X.T, (A - Y)) / len(X)
    db = np.sum(A - Y) / len(X)
    return dW, db

def train_model(X,Y, alpha=0.01) :
    """Generate a Logistical Regression model for a given binary classification training data
    X = Training Data Vector of shape (m,n) [n>0]
    Y = Expected output vector of shape (m,1)
        where,
            m = Training Sample size
            n = Feature size
    """

    if X.shape[1] == None or X.shape[1] == 0 :
        ValueError("Feature dimension (n) is 0")
    if Y.shape[1] == None or Y.shape[1] != 1 :
        ValueError("Invalid output label matrix Y. (Y.shape) = (m,1)")
    if X.shape[0] != Y.shape[0] :
        ValueError("Size (m) of Training Data X and Output Lables Y doesn't match")

    # Initialise parameters
    m, n = X.shape
    cost = 0
    prev_cost = float("inf")
    W = np.zeros(shape=(n,1))
    b = 0

    # Train Model
    for i in range(0, TRAINING_CUT_OFF_LIMIT) :

        A = predict(X, W, b)
        cost = calculate_cost(Y, A)
        dW, db = calculate_cost_derivative(X, Y, A)

        if abs(prev_cost-cost) < 1e-8 :
            print(f"Stopped at iteration {i}, as there is insignificat change in cost.")
            print(f"Weight matrix : {W}\nBias : {b}")
            break

        if np.linalg.norm(dW) < 1e-6 and abs(db) < 1e-6:
            print(f"Stopped at iteration {i}, all the derivatives have flattend")
            print(f"Iteration {i}, Cost={cost}, w={W}, b={b}")
            break

        W = W - (alpha * dW)
        b = b - (alpha * db)
        prev_cost=cost

        if i % 1000 == 0 :
            print(f"Iteration {i} : Cost={cost}\nb = {b} , W = {W}")

    print(f"\nTraining completed...\n")
    print(f"Weights, W = {W}\nBias b = {b}")
    return W, b
