import h5py
import numpy as np
import json
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


file_path = "mini_projects/cat_classifier/models/model_dumps.json"

with open(file_path, "r") as file :
    trained_model_db = json.load(file)
    trained_model_key = trained_model_db["model_keys"][-1]
    trained_model = trained_model_db[trained_model_key]

m = train_X.shape[0]

model = NeuralNetwork(trained_model["layer_dimension"], learning_rate=0.001, max_learning_iterations=100000, min_cost_delta=1e-10)

print("\n"+"--"*30)
print("TRAINING SET ACCURACY")
print("--"*30)

X = train_X
Y = train_Y

Y_predicted = model.predict(X.T, parameters=trained_model["parameters"])

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


print("\n"+"--"*30)
print("TEST SET ACCURACY")
print("--"*30)

X = test_X
Y = test_Y

Y_predicted = model.predict(X.T, parameters=trained_model["parameters"])

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
