import pandas as pd
import h5py
import numpy as np


LOAN_APPROVAL_DATASET_PATH = "dataset/loan_data.csv"


def __normalize_data(col, mean, std):
    return (col - mean) / std


def get_processed_loan_approval_dataset(data_set_path: str = LOAN_APPROVAL_DATASET_PATH, data_normalizaation_enabled = True, training_sample_ratio = 0.8, feature_columns: list = [] ):
    master_data = pd.read_csv("dataset/loan_data.csv", header=0)

    master_data_size = len(master_data)
    training_sample_size = round(training_sample_ratio * master_data_size)

    for col in master_data.select_dtypes("object").columns:
        master_data[col] = master_data[col].astype("category").cat.codes

    label_column = ["loan_status"]
    if len(feature_columns) == 0 :
        feature_columns = list(master_data.keys())
        feature_columns.remove(label_column[0])

    X_train = master_data[feature_columns][:training_sample_size]
    Y_train = master_data[label_column][:training_sample_size]

    X_test = master_data[feature_columns][training_sample_size:]
    Y_test = master_data[label_column][training_sample_size:]

    if data_normalizaation_enabled :
        normalization_parameters = {}
        for col in X_train.keys():
            normalization_parameters[col] = {
                "mean": master_data[col].mean(),
                "std": master_data[col].std(),
            }
            X_train[col] = __normalize_data(
                X_train[col],
                normalization_parameters[col]["mean"],
                normalization_parameters[col]["std"],
            )
            X_test[col] = __normalize_data(
                X_test[col],
                normalization_parameters[col]["mean"],
                normalization_parameters[col]["std"],
            )

    return X_train.to_numpy(), Y_train.to_numpy(), X_test.to_numpy(), Y_test.to_numpy()

def __generate_augmented_image_data(X, Y):
    """
    Generate augmented image data by flipping/roatting the existing image dataset

    Arguments :
        X => numpy array of image dataset of shape (m,i,j,x)
        Y => numpy array of image labels of shape (m,1)
            where,
                m = total number of training samples in the training set
                i,j = dimension of each image
                x = pixel size (3 for RGB, 1 for B/W)

    Return :
        augmented_X => numpy array of image dataset with original and augmented images
        augmented_Y => numpy array of image labels with original and augmented images
    """

    augmented_X = []
    augmented_Y = []

    for i in range(Y.shape[0]):
        augmented_X.append(X[i])
        augmented_Y.append(Y[i][0])
        augmented_X.append(np.flip(X[i], axis=1))
        augmented_Y.append(Y[i][0])
        if Y[i][0] == 1:
            augmented_X.append(np.rot90(X[i], k=1))
            augmented_Y.append(Y[i][0])
            augmented_X.append(np.rot90(X[i], k=-1))
            augmented_Y.append(Y[i][0])

    augmented_X = np.array(augmented_X)
    augmented_Y = np.array(augmented_Y)
    augmented_Y = augmented_Y.reshape(augmented_Y.shape[0], 1)

    return augmented_X, augmented_Y

def get_processed_cat_image_dataset() :
    
    with h5py.File("dataset/cat_data/train_catvsnoncat.h5", "r") as hf:
        train_X_original = hf["train_set_x"][:]
        train_Y_original = hf["train_set_y"][:]
        train_Y_original = train_Y_original.reshape(train_Y_original.shape[0], 1)

    with h5py.File("dataset/cat_data/test_catvsnoncat.h5", "r") as hf:
        test_X_original = hf["test_set_x"][:]
        test_Y_original = hf["test_set_y"][:]
        test_Y_original = test_Y_original.reshape(test_Y_original.shape[0], 1)

    train_X, train_Y = __generate_augmented_image_data(train_X_original, train_Y_original)
    test_X, test_Y = test_X_original, test_Y_original

    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    return train_X, train_Y, test_X, test_Y
