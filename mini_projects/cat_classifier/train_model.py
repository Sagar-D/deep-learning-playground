import h5py
import numpy as np
from n_layer_neural_network import NeuralNetwork

train_X = []
train_Y = []
test_X = []
test_Y = []

with h5py.File("dataset/cat_data/train_catvsnoncat.h5", "r") as hf:
    train_X_original = hf["train_set_x"][:]
    train_X = train_X_original.reshape(train_X_original.shape[0],-1)
    train_Y = hf["train_set_y"][:]
    train_Y = train_Y.reshape(train_Y.shape[0],1)
    print(train_X.shape, train_Y.shape)

with h5py.File("dataset/cat_data/test_catvsnoncat.h5", "r") as hf:
    test_X_original = hf["test_set_x"][:]
    test_X = test_X_original.reshape(test_X_original.shape[0],-1)
    test_Y = hf["test_set_y"][:]
    test_Y = test_Y.reshape(test_Y.shape[0],1)
    print(test_X.shape, test_Y.shape)

m = train_X.shape[0]
layer_dims=[10,6,3,1]

model = NeuralNetwork(layer_dims, learning_rate=0.001, max_learning_iterations=100000, min_cost_delta=1e-10)
model_parameters = model.train(train_X.T, train_Y.T)


for X, Y in zip([train_X, test_X], [train_Y, test_Y]) :

    Y_predicted = model.predict(X.T)
    print(f"\n\nTotal '0' predicted : {np.sum(Y_predicted == 0)}")
    print(f"Total '1' predicted : {np.sum(Y_predicted == 1)}")

    true_positives = np.sum(Y_predicted == Y.T)
    false_positives = np.sum
    prediction_success_count = np.sum(Y_predicted == Y.T)
    prediction_success_rate = (prediction_success_count / Y.T.shape[0]) * 100
    print(f"Model Accuracy : {prediction_success_rate}%")
    


