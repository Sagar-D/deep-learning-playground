import numpy as np
from data_pre_processor import get_processed_dataset
from l2_regularized_weighted_nn import NeuralNetwork

train_X, train_Y, test_X, test_Y = get_processed_dataset()

print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)

print(np.sum(train_Y==1))
print(np.sum(train_Y==0))

m = train_X.shape[0]
layer_dims = [10, 8, 6, 3, 1]

model = NeuralNetwork(
    layer_dims,
    learning_rate=0.01,
    max_learning_iterations=100000,
    min_cost_delta=1e-15,
    enable_validation=False,
    enable_l2_regularization=False
)
model_parameters = model.train(train_X.T, train_Y.T)


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
