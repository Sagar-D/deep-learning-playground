import pandas as pd


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
