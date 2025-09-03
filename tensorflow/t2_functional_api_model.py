import tensorflow as tf
from tensorflow.keras import models, optimizers, layers


input_layer = layers.Input(shape=(784,))

hidden_layer = layers.Dense(128, activation="relu")(input_layer)
skip_layer = hidden_layer
hidden_layer = layers.Dense(128, activation="relu")(hidden_layer)
hidden_layer = layers.add([hidden_layer, skip_layer])
output_layer = layers.Dense(10, activation="softmax")(hidden_layer)

model = models.Model(input_layer, output_layer)

model.compile(
    optimizer = optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=32)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")