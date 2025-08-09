import numpy as np

np.random.seed(42)

# Number of samples
m = 100

# Features
X_train = np.random.rand(m, 3) * 10  # values between 0 and 10

# True weights and bias for synthetic pattern
true_W = np.array([[0.8], [-0.5], [1.2]])
true_b = -4.0


# Logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Generate probabilities from true model
logits = np.dot(X_train, true_W) + true_b
probabilities = sigmoid(logits)

# Labels (with some noise)
Y_train = (
    (probabilities + np.random.normal(0, 0.05, size=probabilities.shape) > 0.5)
    .astype(int)
    .flatten()
)

# Test set
X_test = np.random.rand(20, 3) * 10
logits_test = np.dot(X_test, true_W) + true_b
probabilities_test = sigmoid(logits_test)
Y_test = (
    (
        probabilities_test + np.random.normal(0, 0.05, size=probabilities_test.shape)
        > 0.5
    )
    .astype(int)
    .flatten()
)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Example labels:", Y_train[:10])
