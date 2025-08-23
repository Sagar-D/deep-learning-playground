import numpy as np
import datetime
import json
from plot import plot_cost_curve

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
                 max_learning_iterations = 10000, regularisation_rate=1e-3, save_model_flag = True, enable_validation=True):

        self.learning_rate = learning_rate
        self.min_cost_delta = min_cost_delta
        self.max_learning_iterations = max_learning_iterations
        self.is_model_trained = False
        self.iteration = 0
        self.training_cost_change_data = []
        self.validation_cost_change_data = []
        self.cache = {}
        self.save_model_flag = save_model_flag
        self.l2_lambd = regularisation_rate
        self.enable_validation = enable_validation

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
        
        if type(initial_parameters) == dict:
            self.parameters = initial_parameters
            return

        X = initial_parameters
        if len(self.layer_dims) == self.L :
            self.layer_dims.insert(0,X.shape[0])
        else :
            self.layer_dims[0] = X.shape[0]

        self.parameters = {}
        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = (np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01)
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
            return self.__calculate_activation(Z, "sigmoid") * (1 - self.__calculate_activation(Z, "sigmoid"))
        
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
        derivatives["dZ" + str(self.L)] = self.cache["A" + str(self.L)] - Y
        derivatives["dW" + str(self.L)] = 1/m * np.dot(derivatives["dZ" + str(self.L)], self.cache["A" + str(self.L - 1)].T) + (self.l2_lambd/m) * parameters["W"+str(self.L)]
        derivatives["db" + str(self.L)] = 1/m * np.sum(derivatives["dZ" + str(self.L)], axis=1, keepdims=True)

        for l in range(self.L - 1, 0, -1):
            derivatives["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, derivatives["dZ" + str(l + 1)]) * self.__derivative_of_activation(self.cache["Z" + str(l)], "relu")
            derivatives["dW" + str(l)] = 1/m * np.dot(derivatives["dZ" + str(l)], self.cache["A" + str(l - 1)].T) + (self.l2_lambd/m) * parameters["W"+str(l)]
            derivatives["db" + str(l)] = 1/m * np.sum(derivatives["dZ" + str(l)], axis=1, keepdims=True)

        return derivatives


    def __optimize(self, parameters, derivatives) :
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
            parameters["W"+str(l)] -= self.learning_rate * derivatives["dW"+str(l)]
            parameters["b"+str(l)] -= self.learning_rate * derivatives["db"+str(l)]
        
        return parameters

    def train(self, X, Y, initail_parameters=None, X_valid=None, Y_valid=None):
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
        
        initail_parameters = initail_parameters if initail_parameters != None and type(initail_parameters) == dict else X
        self.__init_parameters(initail_parameters)

        for self.iteration in range(self.max_learning_iterations) :

            AL = self.__forward_propagate(X, self.parameters)
            cost = self._calculate_cost(AL, Y, parameters=self.parameters)
            derivatives = self.__back_propagate(Y, self.parameters)

            if abs(prev_cost-cost) < self.min_cost_delta :
                print(f"Stop learning at iteration {self.iteration}. No signinficant change in the cost {cost}")
                break

            self.parameters = self.__optimize(self.parameters, derivatives)
            prev_cost = cost

            if self.iteration % 100 == 0 :
                self.training_cost_change_data.append(cost)
                if self.enable_validation and isinstance(X_valid, np.ndarray) and isinstance(Y_valid, np.ndarray) :
                    Y_predicted = self.predict(X_valid)
                    validation_cost = self._calculate_cost(Y_predicted, Y_valid, self.parameters)
                    self.validation_cost_change_data.append(validation_cost)
                print(f"Iteration : {self.iteration} | Cost : {cost} | Validation Cost : {'NA' if len(self.validation_cost_change_data) == 0 else self.validation_cost_change_data[-1] }")

            if self.iteration % 10000 == 0 :
                plot_cost_curve(self.training_cost_change_data, self.validation_cost_change_data)

        print("\n\nModel treaining completed!!!!\n")
        self.is_model_trained = True
        self.save_model()

        for l in range(1, self.L + 1) :
            print(f"W{l} : {self.parameters['W'+str(l)]} \nB{l} : {self.parameters['b'+str(l)]}")

        return self.parameters


    def predict(self, X, parameters=None) :
        parameters = parameters if parameters else self.parameters
        A = self.__forward_propagate(X, parameters)
        return (A > 0.5).astype(int)
    
    def save_model(self) :

        if not self.save_model_flag :
            print("Save Model feature is turned off. Not saving the model to model_dump.json")
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
            file_path = "mini_projects/cat_classifier/models/model_dumps.json"
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

