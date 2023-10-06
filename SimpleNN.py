import numpy as np
import sys
import csv
import random

import yaml
from matplotlib import pyplot as plt


class simpleNN:
    def __init__(self, input_size=2, hidden_size=3, output_size=1, epochs=10, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.params = {}
        self.loss = []

    def initialize_weights(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def relu(self, Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def read_input_data(self, filename):
        '''
        Reads input data (train or test.csv) from the CSV file.
        Parameters:
            filename - CSV file name (string)
                CSV file format:
                    input1, input2, ..., output
                                    ...
                                    ...
            normalize - flag for data normalization (bool, optional)
        Sets:
            self.Nin = number of inputs of the perceptron (int)
        Returns:
            X - input training data (list)
            Y - output (expected) training data (list)
        '''

        # Read CSV data
        try:
            file = open(filename, 'rt')
        except FileNotFoundError:
            sys.exit('Error: data file does not exists.')

        dataset = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        # Construct the X and Y lists. This is a simple perceptron with only one output,
        # so X should contain all data from all columns except the last one,
        # and Y - data from the last column only.
        X = []
        Y = []

        try:
            for line in dataset:
                list = []
                X.append(line[0:-1])
                list.append(line[-1])
                Y.append(list)
        except ValueError:
            sys.exit('Error: Wrong format of the CSV file.')

        file.close()

        if self.input_size == 0:
            sys.exit('Error: zero-length training vector.')

        return np.array(X), np.array(Y)

    def entropy_loss(self, y, ypred):
        sumWeighted = 0
        for i in range(len(y)):
            sumWeighted += (ypred[i] - y[i]) ** 2
        return sumWeighted

    def forward_propagation(self, Xtrain, Ytrain):
        Z1 = np.dot(Xtrain, self.W1)
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2)
        ypred = self.relu(Z2)

        self.loss.append(np.sqrt(self.entropy_loss(Ytrain, ypred))/len(Xtrain))

        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        self.params['A2'] = ypred

        return ypred

    def back_propagation(self, Xtrain, Ytrain, ypred):

        grad_A2 = 2 * (self.params['A2'] - Ytrain) / len(Ytrain)
        grad_W2 = self.params['A1'].T.dot(grad_A2)
        grad_A1 = grad_A2.dot(self.W2.T)
        grad_W1 = Xtrain.T.dot(grad_A1)

        self.W1 -= self.learning_rate * grad_W1
        self.W2 -= self.learning_rate * grad_W2

    def train(self, Xtrain, Ytrain):

        self.initialize_weights()

        for epoch in range(self.epochs):
            print('Epoch = {}'.format(epoch + 1))

            ypred = self.forward_propagation(Xtrain, Ytrain)
            self.back_propagation(Xtrain, Ytrain, ypred)

            print('Loss = {}'.format(self.loss[epoch]))

        self.plot_loss()


    def test(self, Xtest):
        Z1 = Xtest.dot(self.W1)
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.W2)
        pred = self.relu(Z2)

        return pred

    def plot_loss(self, filename='loss.png'):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Loss curve for training")
        plt.savefig(filename)
        plt.show()
        print('Plot has been saved to the file', filename)

    def save_model(self, filename='sum_model.model'):
        '''
        Saves the perceptron data into a file.
        Parameters:
            filename - file name (str)
        Returns:
            None
        '''
        data = {"Nin": self.input_size,
                "Nhid": self.hidden_size,
                "Nout": self.output_size,
                "Epochs": self.epochs,
                "LearningRate": self.learning_rate,
                "W1": self.W1.tolist(),
                "W2": self.W2.tolist()}

        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
            # yaml.dump(self.min_val, outfile, default_flow_style=False)

        print('Model saved to file', filename)

    def load_model(self, filename):
        '''
        Loads the perceptron data from a file.
        Parameters:
            filename - file name (str)
        Returns:
            None
        '''

        with open(filename) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            self.input_size = data['Nin']
            self.hidden_size = data['Nhid']
            self.output_size = data['Nout']
            self.epochs = data['Epochs']
            self.learning_rate = data['LearningRate']
            self.W1 = np.array(data['W1'])
            self.W2 = np.array(data['W2'])

        print('Model loaded from file', filename)

