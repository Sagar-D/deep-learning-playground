import h5py
import numpy as np
from l2_regularized_neural_network import NeuralNetwork
from data_augmentation import generate_augmented_image_data

with h5py.File("dataset/cat_data/train_catvsnoncat.h5", "r") as hf:
    train_X_original = hf["train_set_x"][:]
    train_Y_original = hf["train_set_y"][:]
    train_Y_original = train_Y_original.reshape(train_Y_original.shape[0], 1)

with h5py.File("dataset/cat_data/test_catvsnoncat.h5", "r") as hf:
    test_X_original = hf["test_set_x"][:]
    test_Y_original = hf["test_set_y"][:]
    test_Y_original = test_Y_original.reshape(test_Y_original.shape[0], 1)

train_X, train_Y = generate_augmented_image_data(train_X_original, train_Y_original)
test_X, test_Y = test_X_original, test_Y_original

train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

m = train_X.shape[0]
layer_dims = [10, 6, 3, 1]

model = NeuralNetwork(
    layer_dims,
    learning_rate=0.001,
    max_learning_iterations=100000,
    min_cost_delta=1e-15,
    enable_validation=True
)
model_parameters = model.train(train_X.T, train_Y.T, X_valid=test_X.T, Y_valid=test_Y.T)


for X, Y in zip([train_X, test_X], [train_Y, test_Y]):

    Y_predicted = model.predict(X.T)

    accurate_predictions_count = np.sum(Y_predicted == Y.T)
    true_positives = np.sum((Y_predicted == 1) & (Y.T == 1))
    true_negatives = np.sum((Y_predicted == 0) & (Y.T == 0))
    false_positives = np.sum((Y_predicted == 1) & (Y.T == 0))
    false_negatives = np.sum((Y_predicted == 0) & (Y.T == 1))

    print(f"True Positives : {true_positives}")
    print(f"True Negatives : {true_negatives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"\nModel Accuracy : {(true_positives+true_negatives)/Y_predicted.shape[1]}")
    print(f"Model Precision [TP/(TP+FP)] : {true_positives/(true_positives + false_positives)}")
    print(f"Model Recall [TP/(TP+FN)] : {true_positives/(true_positives + false_negatives)}")
