import numpy as np
import datetime
import json

class NeuralNetwork() :
    """
    Create, train and inference a n-layer nueral network model.
    
    Arguments :
        layer_dimension => A tuple representing number of neurons in each ayer of the neural network
        learning_rate => Learning rate (alpha) for training the model
        min_cost_delta => Minimum cost delta between two iterations, below which training should be terminated.
        max_learning_iterations => Maximum number of iterations of training to be performed.
    """

    def __init__(self,layer_dimension: tuple, learning_rate = 0.01, min_cost_delta = 1e-10, 
                 max_learning_iterations = 10000, enable_l2_regularization=True, regularisation_rate=1e-3, save_model_flag = True, save_model_path="", enable_validation=True, momentum_window_beta = 0.9):

        self.learning_rate = learning_rate
        self.min_cost_delta = min_cost_delta
        self.max_learning_iterations = max_learning_iterations
        self.is_model_trained = False
        self.iteration = 0
        self.training_cost_change_data = []
        self.validation_cost_change_data = []
        self.cache = {}
        self.save_model_flag = save_model_flag
        self.save_model_path = save_model_path
        self.enable_validation = enable_validation
        self.enable_l2_regularization = enable_l2_regularization
        self.l2_lambd = regularisation_rate
        self.momentum_window_beta = momentum_window_beta

        self.layer_dims = [*layer_dimension]
        self.L = len(layer_dimension)


    def __init_parameters(self, initial_parameters: dict) :
        """
        Initialize the parameters to a custom value. Can be used for model fine tuning.
        Note: Make sure that parametrs dict has weights and biases for all layers.
        
        Arguments :
            initial_parameters => Dictionary of Weights and biases for all the layers.
                Note : if initial_parameters is empty or not a dictionary, then parameters will be initialized randonly.
        """
        if self.is_model_trained :
            raise RuntimeWarning("You are trying to initialize parameters after training the model. This can overwrite the trained model parameters!!!")
        
        if type(initial_parameters) == dict and initial_parameters:
            self.parameters = initial_parameters
            return

        X = initial_parameters
        if len(self.layer_dims) == self.L :
            self.layer_dims.insert(0,X.shape[0])
        else :
            self.layer_dims[0] = X.shape[0]

        self.parameters = {}
        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * np.sqrt(2. / self.layer_dims[l-1])
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))


    def __calculate_activation(self, Z, activation_method="relu"):
        """
        Perform activation function and return result
        
        Arguments:
            Z => Vector of shape (n,m)
            activation_method => Activation function name. Supports : (sigmoid, relu)
        
        Return :
            A => Activation Vector
        """
        activation_method = activation_method.strip().lower()
        if activation_method == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        
        if activation_method == "relu":
            return np.maximum(0, Z)

        raise ValueError(f"Unsupported activation method {activation_method} passsed as argument. Supported methods - (sigmoid, relu)")


    def __forward_propagate(self, X, parameters):
        """
        Perform forward propagation of neural betwork to predict yhat
        
        Arguments :
            X => Input vector of shape (n,m)
                where,
                    n is number of features in training sample
                    m is number of training sampels in the data set
            parameters => dictionary of weights and bias for every layer of the network
        
        Return :
            AL => Activation output of final layer (Yhat)
        """
        self.cache = {"A0": X}
        for l in range(1, self.L+1) :
            activation_method = "sigmoid" if l == self.L else "relu"
            self.cache["Z" + str(l)] = np.dot(parameters["W" + str(l)], self.cache["A" + str(l - 1)]) + parameters["b" + str(l)]
            self.cache["A" + str(l)] = self.__calculate_activation(self.cache["Z" + str(l)], activation_method) 
        
        AL = self.cache["A" + str(self.L)]
        return AL


    def _calculate_cost(self, A, Y, parameters):
        """
        Calculate cost of predicted values A w.r.t expected Values Y
        
        Arguments:
            Y => Expected output vector of shape (1,m)
            A => Predicted output vector of shape (1,m)
        
        Returns:
            cost => Log Loss value
        """
        m = A.shape[1]
        epsilon = 1e-7
        A = np.clip(A, epsilon, 1 - epsilon)
        cost = -1 / m * np.sum(Y * np.log(A) + ((1 - Y) * np.log(1 - A)))
        
        if self.enable_l2_regularization :
            l2 = 0
            for l in range(1, self.L+1) :
                l2 += np.sum((parameters["W"+str(l)] * parameters["W"+str(l)]))
            cost += (self.l2_lambd/(2*m)) * l2

        return cost


    def __derivative_of_activation(self, Z, activation_method):
        """
        Compute the derivative of the activation function
        
        Arguments :
            Z => Vector used as input for activation
            activation_method => activation method to be used. Supports : (sigmoid, relu)
        
        Return :
            derivative_of_activation_function
        """
        activation_method = activation_method.strip().lower()
        if activation_method == "sigmoid":
            sigmoid_Z = self.__calculate_activation(Z, "sigmoid")
            return sigmoid_Z * (1 - sigmoid_Z)
        
        if activation_method == "relu":
            return np.where(Z>0, 1, 0)
        
        raise ValueError(f"Unsupported activation method {activation_method} passsed as argument. Supported methods - (sigmoid, relu)")


    def __back_propagate(self, Y, parameters):
        """
        Perform backward propagation of neural betwork to compute derivatives
        
        Arguments :
            parameters => dictionary of weights and bias for every layer of the network
            layer_dims => List of integers representing number of neurons in each layer of the network.
        
        Return :
            derivatives => Dictionary of partial derivatives of Cost function w.r.t Weights and biases for every layer
        """
        
        m = Y.shape[1]
        derivatives = {}
        AL = self.cache["A" + str(self.L)]
        derivatives["dZ" + str(self.L)] = AL - Y
        derivatives["dW" + str(self.L)] = 1/m * np.dot(derivatives["dZ" + str(self.L)], self.cache["A" + str(self.L - 1)].T)
        if self.enable_l2_regularization :
            derivatives["dW" + str(self.L)] += (self.l2_lambd/m) * parameters["W"+str(self.L)]
        derivatives["db" + str(self.L)] = 1/m * np.sum(derivatives["dZ" + str(self.L)], axis=1, keepdims=True)

        for l in range(self.L - 1, 0, -1):
            derivatives["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, derivatives["dZ" + str(l + 1)]) * self.__derivative_of_activation(self.cache["Z" + str(l)], "relu")
            derivatives["dW" + str(l)] = 1/m * np.dot(derivatives["dZ" + str(l)], self.cache["A" + str(l - 1)].T)
            if self.enable_l2_regularization :
                derivatives["dW" + str(l)] += (self.l2_lambd/m) * parameters["W"+str(l)]
            derivatives["db" + str(l)] = 1/m * np.sum(derivatives["dZ" + str(l)], axis=1, keepdims=True)

        return derivatives


    def __optimize(self, parameters, momentum_derivatives) :
        """
        Optimize the parameters for model training.

        Arguments :
            parameters => dictionary of weights and bias for every layer of the network
            derivatives => Dictionary of partial derivatives of Cost function w.r.t Weights and biases for every layer
            learning_rate => learning rate

        Return :
            parameters => updated dictionary of optimized weights and bias for every layer of the network
        """
        for l in range(1, self.L+1) :
            parameters["W"+str(l)] -= self.learning_rate * momentum_derivatives["dW"+str(l)]
            parameters["b"+str(l)] -= self.learning_rate * momentum_derivatives["db"+str(l)]
        
        return parameters

    def train(self, X, Y, batch_size= 256, initail_parameters=None, X_valid=None, Y_valid=None):
        """
        Train a neural network model to fit given training samples.

        Arguments :
            X => Input feature vector of shape (n,m)
            Y => Output label vector of shape (1,m)
                where,
                    n is number of features in training sample
                    m is total number of training samples in the training set
            initial_parameters => A dictionary of weights and biases fro every layer to initialise the training
        """
        cost = 0
        prev_cost = float('inf')
        curr_batch_index = 0


        initail_parameters = initail_parameters if initail_parameters != None and type(initail_parameters) == dict else X
        self.__init_parameters(initail_parameters)

        momentum_derivatives = {}
        for l in range(1, len(self.layer_dims)):
            momentum_derivatives['dW'+str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l - 1]))
            momentum_derivatives["db" + str(l)] = np.zeros((self.layer_dims[l], 1))

        for self.iteration in range(self.max_learning_iterations) :

            # shuffling / rearranging the training samples to avoid model from 
            # learning patterns in the ordering of datase
            permutaions = np.random.permutation(X.shape[1])
            X = X[:, permutaions]
            Y = Y[:, permutaions]

            # Train on the mini-batch samples
            X_batch = X.T[curr_batch_index: curr_batch_index + batch_size].T
            Y_batch = Y.T[curr_batch_index: curr_batch_index + batch_size].T

            AL = self.__forward_propagate(X_batch, self.parameters)
            cost = self._calculate_cost(AL, Y_batch, parameters=self.parameters)
            derivatives = self.__back_propagate(Y_batch, self.parameters)

            if abs(prev_cost - cost) < self.min_cost_delta :
                print(f"Stop learning at iteration {self.iteration}. No signinficant change in the cost {cost}")
                break

            for l in range(1, len(self.layer_dims)):
                momentum_derivatives['dW'+str(l)] = self.momentum_window_beta * momentum_derivatives['dW'+str(l)] + (1-self.momentum_window_beta) * derivatives['dW'+str(l)]
                momentum_derivatives['db'+str(l)] = self.momentum_window_beta * momentum_derivatives['db'+str(l)] + (1-self.momentum_window_beta) * derivatives['db'+str(l)]
            
            self.parameters = self.__optimize(self.parameters, momentum_derivatives)
            prev_cost = cost

            curr_batch_index += batch_size
            if curr_batch_index + batch_size >= X.shape[1]:
                curr_batch_index = 0

            if self.iteration % 100 == 0 :
                self.training_cost_change_data.append(cost)
                if self.enable_validation and isinstance(X_valid, np.ndarray) and isinstance(Y_valid, np.ndarray) :
                    Y_predicted = self.predict(X_valid)
                    validation_cost = self._calculate_cost(Y_predicted, Y_valid, self.parameters)
                    self.validation_cost_change_data.append(validation_cost)
                print(f"Iteration : {self.iteration} | Cost : {cost} | Validation Cost : {'NA' if len(self.validation_cost_change_data) == 0 else self.validation_cost_change_data[-1] }")

        print("\n\nModel treaining completed!!!!\n")
        self.is_model_trained = True
        self.save_model(self.save_model_path)

        for l in range(1, self.L + 1) :
            print(f"W{l} : {self.parameters['W'+str(l)]} \nB{l} : {self.parameters['b'+str(l)]}")

        return self.parameters


    def predict(self, X, parameters=None) :
        parameters = parameters if parameters else self.parameters
        A = self.__forward_propagate(X, parameters)
        return (A > 0.5).astype(int)
    
    def save_model(self, file_path="") :

        if not self.save_model_flag :
            print("Save Model feature is turned off. Not saving the model to model_dump.json")
            return
        
        if type(file_path) != str or file_path.strip() == "" :
            print("Valid file_path is not passed to save the model. Skipping this step!")
            return

        current_timestamp = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        parameters_serializable = {}
        for key, value in self.parameters.items() :
            parameters_serializable[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        model = {
            "created_at": current_timestamp,
            "parameters": parameters_serializable,
            "layer_dimension": self.layer_dims[1:],
            "learning_rate": self.learning_rate,
            "l2_regularization_lambda": self.l2_lambd,
            "learning_iterations_executed": self.iteration,
            "min_cost_delta": self.min_cost_delta,
            "training_cost_change_data": self.training_cost_change_data,
            "validation_cost_change_data": self.validation_cost_change_data
        }

        try :
            with open(file_path, "r+") as file:
                content = file.read().strip()
                file.seek(0)
                model_db = json.loads(content) if content else {}
            model_db[current_timestamp] = model
            model_db["model_keys"] = [] if "model_keys" not in model_db.keys() else model_db["model_keys"]
            model_db["model_keys"].append(current_timestamp)
            with open(file_path, "w") as file :
                json.dump(model_db, file)
        except Exception as e :
            print(f"An error occured while saving model details to file : {e}")
        
        print(f"Model Saved at location : {file_path}")



if __name__ == "__main__" :

    from data_pre_processor import get_processed_loan_approval_dataset

    X_train, Y_train, X_test, Y_test = get_processed_loan_approval_dataset()

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    m = X_train.shape[0]
    layer_dims = [10, 6, 3, 1]

    model = NeuralNetwork(
        layer_dims,
        learning_rate=0.001,
        max_learning_iterations=20000,
        min_cost_delta=1e-15,
        enable_validation=False,
        enable_l2_regularization=False
    )
    model_parameters = model.train(X_train.T, Y_train.T, batch_size=64)

    ### Test Model ###
    print("....Testing model....\n")

    Y_predicted = model.predict(X_test.T)

    true_positives = np.sum((Y_predicted == 1) & (Y_test.T == 1))
    true_negatives = np.sum((Y_predicted == 0) & (Y_test.T == 0))
    false_positives = np.sum((Y_predicted == 1) & (Y_test.T == 0))
    false_negatives = np.sum((Y_predicted == 0) & (Y_test.T == 1))

    print(f"True Positives : {true_positives}")
    print(f"True Negatives : {true_negatives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"\nModel Accuracy : {(true_positives+true_negatives)/Y_predicted.shape[1]}")
    print(f"Model Precision [TP/(TP+FP)] : {true_positives/(true_positives + false_positives)}")
    print(f"Model Recall [TP/(TP+FN)] : {true_positives/(true_positives + false_negatives)}")