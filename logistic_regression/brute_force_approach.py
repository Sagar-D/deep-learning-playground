import numpy as np
from typing import List
from time import sleep


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(x, y, w, b):

    if len(x) != len(y):
        raise ValueError("Vectors lenth mismatch!!!")
    a = []
    for i in range(0, len(x)):
        y = float((w * x[i]) + b)
        a.append(sigmoid(y))
    return a


def calculate_cost(y, a):
    if len(y) != len(a):
        raise ValueError("Vectors length mismatch!!!")

    epsilon = 1e-15
    cummulative_loss = 0
    for i in range(len(y)):
        ai = max(min(a[i], 1 - epsilon), epsilon)  # clip to avoid log(0)
        loss = -1 * (y[i] * np.log(ai) + (1 - y[i]) * np.log(1 - ai))
        cummulative_loss += loss

    return cummulative_loss / len(y)


def derivative_of_cost(x, y, a):

    if (len(x) != len(y)) or (len(y) != len(a)):
        raise ValueError("Vectors lenth mismatch!!!")

    cummulative_dLw = 0
    cummulative_dLb = 0
    for i in range(0, len(y)):
        dLw = (a[i] - y[i]) * x[i]
        cummulative_dLw = cummulative_dLw + dLw
        dLb = a[i] - y[i]
        cummulative_dLb = cummulative_dLb + dLb

    return cummulative_dLw / len(y), cummulative_dLb / len(y)


def create_model(x, y):

    MAX_LEARING_REP = 1000000
    w = 0.20
    b = 0.10
    alpha = 0.001
    cost = 0
    prev_cost = float("inf")

    for i in range(0, 10000000):

        a = predict(x, y, w, b)
        cost = calculate_cost(y, a)
        dw, db = derivative_of_cost(x, y, a)

        # Stopping condition
        if abs(prev_cost - cost) < 1e-7:
            print(f"Stopped at iteration {i}, cost change small")
            print(f"Iteration {i}, Cost={cost}, w={w}, b={b}")
            break
        if abs(dw) < 1e-6 and abs(db) < 1e-6:
            print(f"Stopped at iteration {i}, gradients near zero")
            print(f"Iteration {i}, Cost={cost}, w={w}, b={b}")
            break

        w = w - (alpha * dw)
        b = b - (alpha * db)
        prev_cost = cost

        if i % 1000 == 0:
            print(f"Iteration {i}, Cost={cost}, w={w}, b={b}")

    print(f"Model Trained with w = {w} , b = {b}")

    return w, b


x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

x_test = np.array([1.5, 2.5, 4.5, 5.5, 7.5])
y_test = np.array([0, 0, 0, 1, 1])

w, b = create_model(x_train, y_train)

print("\n\n......Running Tests.........")
print("--" * 30)
print("\n")

for x, y in zip(x_test, y_test):
    y_predicted = round(sigmoid(float((w * x) + b)))
    print(f"Predicted Y = {y_predicted} \t| Expected Y = {y}")
