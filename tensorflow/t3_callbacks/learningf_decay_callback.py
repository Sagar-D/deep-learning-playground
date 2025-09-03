import tensorflow as tf
from tensorflow.keras import models, optimizers, layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

lerning_decay_callback = ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 2,
    min_lr=1e-5 
)

model = models.Sequential(
    [
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ]
)

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-3),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    callbacks=[lerning_decay_callback],
)
