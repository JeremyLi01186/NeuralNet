#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd



class NeuralNet:
    def __init__(self, train,activation, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train,header=None)

        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        #train_dataset = raw_input
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X

        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self,x)
        # MY CODE
        if activation == "tanh":
            self.__tanh(self,x)
        if activation == "Relu":
            self.__relu(self,x)
    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "Relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #my code
    def __tanh(self,x):
        return 2/(1 + np.exp(-2*x)) - 1


    def __relu(self,x):
        for index, value in np.ndenumerate(x):
            if value > 0:
                x[index] = value
            elif value <= 0:
                x[index] = 0
        return x


    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # my code
    def __tanh_derivative(self,x):
        return 1 - np.power(x,2)


    def __relu_derivative(self,x):
        for index, value in np.ndenumerate(x):
            if value > 0:
                x[index] = 1.0
            elif value <= 0:
                x[index] = 0.0
        return x

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        #my code categorical to numerical
        for row in range(len(X[0])):
            if X.iloc[row,-1] == "Iris-setosa":
                X.iloc[row,-1] = 1
            elif X.iloc[row,-1] == "Iris-versicolor":
                X.iloc[row,-1] = 2
            elif X.iloc[row,-1] == "Iris-virginica":
                X.iloc[row,-1] = 3

        #my code normalization
        X_norm = (X.iloc[:, 0:4] - X.iloc[:, 0:4].min()) / (X.iloc[:, 0:4].max() - X.iloc[:, 0:4].min())


        return X_norm
    # Below is the training function

    def train(self, activation, max_iterations =1000, learning_rate = 0.01):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            #print(out)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input
        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            return out
            # my code
        elif activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)

            return out

        elif activation == "Relu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)

            return out


    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        # my code
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        elif activation == "Relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        # my code
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        elif activation == "Relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        # my code
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        elif activation == "Relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))

        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        elif activation == "Relu":
            delta_input_layer = np.multiply(self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function


    def predict(self,test,activation,header = False):
        # my code
        raw_input = pd.read_csv(test,header=None)
        for row in range(len(raw_input[0])):
            if raw_input.iloc[row, -1] == "Iris-setosa":
                raw_input.iloc[row, -1] = 1
            elif raw_input.iloc[row, -1] == "Iris-versicolor":
                raw_input.iloc[row, -1] = 2
            elif raw_input.iloc[row, -1] == "Iris-virginica":
                raw_input.iloc[row, -1] = 3
        X_norm = (raw_input.iloc[:, 0:4] - raw_input.iloc[:, 0:4].min()) / (raw_input.iloc[:, 0:4].max() - raw_input.iloc[:, 0:4].min())
        train_dataset = X_norm
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        self.y = train_dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        h1=4
        h2 =2
        #self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X

        self.delta01 = np.zeros((input_layer_size, h1))
        #self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        #self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))



        o = self.forward_pass(activation)
        #print(o)
        error = 0.5 * np.power((o - self.y), 2)
        print("testing total error is: " + str(sum(error)))


if __name__ == "__main__":
    neural_network = NeuralNet("iristraining.csv", "tanh")
    neural_network.train("tanh")
    testError = neural_network.predict("iristest.csv", "tanh")


