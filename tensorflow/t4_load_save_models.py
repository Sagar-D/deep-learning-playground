import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

model = models.load_model("tensorflow/models/best_modelcallback.keras")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=32)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

model.save("tensorflow/models/t4_load_save_model.keras")
