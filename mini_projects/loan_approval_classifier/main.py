import numpy as np
import pandas as pd
from binary_classifier import train_model, predict


master_data = pd.read_csv("dataset/loan_data.csv", header=0)
master_data_size = len(master_data)
training_sample_size = round(0.7 * master_data_size)
validation_sample_size = round(0.1 * master_data_size)
testing_sample_size = round(0.2 * master_data_size)

# feature_columns = ["person_age", "person_gender", "person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length","credit_score","previous_loan_defaults_on_file"]
feature_columns = ["loan_percent_income", "previous_loan_defaults_on_file", "credit_score","previous_loan_defaults_on_file"]
label_column = ["loan_status"]
gender_map = { "male": 0, "female": 1}

master_data["person_gender"] = master_data["person_gender"].apply(lambda gender : gender_map[gender])
master_data["previous_loan_defaults_on_file"] = master_data["previous_loan_defaults_on_file"].apply(lambda data : 1 if data.lower() == "yes" else 0)
master_data['credit_score'] = (master_data['credit_score'] - master_data['credit_score'].mean()) / master_data['credit_score'].std()


# X_train = master_data[feature_columns][:10000].to_numpy()
# Y_train = master_data[label_column][:10000].to_numpy()

X_train = master_data[feature_columns][:training_sample_size].to_numpy()
Y_train = master_data[label_column][:training_sample_size].to_numpy()

X_valid = master_data[feature_columns][training_sample_size:training_sample_size+validation_sample_size].to_numpy()
Y_valid = master_data[label_column][training_sample_size:training_sample_size+validation_sample_size].to_numpy()

X_test = master_data[feature_columns][training_sample_size+validation_sample_size:].to_numpy()
Y_test = master_data[label_column][training_sample_size+validation_sample_size:].to_numpy()

print(X_train.shape, Y_train.shape)
W, b = train_model(X_train,Y_train)
print("Model Trained....\n\n")
print(f"W = {W}\n\nb={b}")


### Test Model ###
print("....Testing model....\n")

validation_m = 1000
Y_predicted = np.round(predict(X_valid[:validation_m], W, b))
for yp, y in zip(Y_predicted, Y_valid[:10]) :
    print(f"Predicted = {int(yp)} , Actual = {int(y)}")

model_success_in_count = np.sum(Y_valid[:validation_m] == Y_predicted)
model_success_in_percentage = int(model_success_in_count * 100 / len(Y_predicted))

print("\n", "--" * 30, "\n\n", sep="")
print(f"Prediction Success Rate : {model_success_in_percentage}%")
