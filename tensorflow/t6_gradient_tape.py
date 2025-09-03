import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import optimizers
import numpy as np

### Implementation of logistic regression using gradient tape ###

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [1]], dtype=np.float32)

W = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=1), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

print(X.shape)
print(Y.shape)

def forward_pass(X) :
    return tf.sigmoid(tf.matmul(X,W) + b)

def compute_cost(Y, A) :
    return tf.reduce_mean(binary_crossentropy(Y, A))

optimizer = optimizers.Adam(learning_rate = 1e-1)

for step in range(1000) :
    with tf.GradientTape() as tape :
        A = forward_pass(X)
        cost = compute_cost(Y, A)
        grads = tape.gradient(cost, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))

    if step % 200 == 0:
        print(f"Step {step}, Loss: {cost.numpy():.4f}")


print("Predictions:", (forward_pass(X).numpy() > 0.5).astype(int))
