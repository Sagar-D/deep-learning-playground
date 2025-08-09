import numpy as np
from random import random
from logistic_regression.training_data_generator import X_train, Y_train, X_test, Y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, W, b):
    """Predict the expected outputs using logistic regression
    X : Feature vector of shape (m,n)
    W : Weight Matrix of shape (n,1)
    b : Bias value
    where,
        m = Size of Training Data
        n = Number of Features per training sample
    """
    A = sigmoid(np.dot(X, W) + b)
    return A


def calculat_cost(Y, A):
    """Calculate the cost of logistic regression
    Y : Expected Output vector of shape (m,1)
    A : Predicted Output vector of shape (m,1)
    where,
        m = Size of Training Data
        n = Number of Features per training sample
    """

    def handle_ln_zero(M):
        epsilon = 1e-15
        return np.maximum(np.minimum(M, 1 - epsilon), epsilon)

    log_A = np.log(handle_ln_zero(A))
    log_Aminus1 = np.log(handle_ln_zero(1 - A))
    L = -1 * ((Y * log_A) + ((1 - Y) * (log_Aminus1)))
    return np.mean(L)


def derivative_of_cost(X, Y, A):
    """Calculate the derivative of cost function of logistic regression
    X : Feature vector of shape (m,n)
    Y : Expected Output vector of shape (m,1)
    A : Predicted Output vector of shape (m,1)
    where,
        m = Size of Training Data
        n = Number of Features per training sample
    """
    dW = np.dot(X.T, (A - Y)) / len(Y)
    db = np.sum(A - Y) / len(Y)
    return dW, db


def create_model(X, Y):
    """Create a Logistic Regression Model.
    X : Feature vector of shape (m,n)
    Y : Expected Output vector of shape (m,1)
    where,
        m = Size of Training Data
        n = Number of Features per training sample
    """

    MAX_LEARNING_REP = 1000000
    X = X
    n = X.shape[1]
    W = np.array([random() for x in range(0, n)]).reshape(-1, 1)
    b = random()
    alpha = 0.001
    cost = 0
    prev_cost = float("inf")

    for i in range(0, MAX_LEARNING_REP):

        A = predict(X, W, b)
        cost = calculat_cost(Y, A)
        dW, db = derivative_of_cost(X, Y, A)

        # Stopping condition
        if np.linalg.norm(dW) < 1e-6 and abs(db) < 1e-6:
            print(f"Stopped at iteration {i}, cost change small")
            print(f"Iteration {i}, Cost={cost}, w={W}, b={b}")
            break

        if abs(prev_cost - cost) < 1e-7:
            print(f"Stopped at iteration {i}, cost change small")
            print(f"Iteration {i}, Cost={cost}, w={W}, b={b}")
            break

        W = W - (alpha * dW)
        b = b - (alpha * db)
        prev_cost = cost

        if i % 1000 == 0:
            print(f"Iteration {i} : Cost={cost}\n")

    print(f"Model Trained with w = {W} , b = {b}")
    return W, b


if __name__ == "__main__":

    X = X_train
    Y = Y_train.reshape(X.shape[0], -1)
    W, b = create_model(X, Y)
    print(f"Weights : {W}\nBias : {b}\n\n")

    ### Test Model ###

    print("....Testing model....\n")
    Y_predicted = sigmoid((np.dot(X_test, W)) + b)
    Y_predicted = np.round(Y_predicted)
    Y_predicted = Y_predicted.astype(int)
    Y_predicted = Y_predicted.reshape(-1)

    for yp, y in zip(Y_predicted, Y_test):
        print(f"Expected Y : {y} \t | Predicted Y : {yp}")

    model_success_in_count = np.sum(Y_test == Y_predicted)
    model_success_in_percentage = int(model_success_in_count * 100 / len(Y_predicted))

    print("\n", "--" * 30, "\n\n", sep="")
    print("Predictions\t:", Y_predicted)
    print("Actual\t\t:", Y_test)
    print(f"Prediction Success Rate : {model_success_in_percentage}%")
