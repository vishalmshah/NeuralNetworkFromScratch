import math
import numpy as np
import pandas as pd
from pprint import pprint

class Network:

    # Initializes Network class
    # Creates network layers, weights, and biases
    # Params: size of input and output layers
    def __init__(self, n_input, n_output):
        # Number of layers including input and output layers
        # input layer is = to number of columns in dataframe
        self.layers = [ np.zeros(shape=(n_input, 1)), np.zeros(shape=(16,1)), np.zeros(shape=(16,1)), np.zeros(shape=(n_output,1)) ]

        # array of weights between each of the 4 layers (2 hidden layers), length is n-1
        # w_jk is how we index these (j is second layer, k is first)
        # We start with random values to allow for the neurons to differentiate better
        self.weights = [ np.random.rand(len(self.layers[1]), len(self.layers[0])), 
                    np.random.rand(len(self.layers[2]), len(self.layers[1])), 
                    np.random.rand(len(self.layers[3]), len(self.layers[2])) ]
        # array of biases for each neuron
        # again, start with random values
        self.biases = [ np.random.rand(len(self.layers[1]), 1),
                np.random.rand(len(self.layers[2]), 1),
                np.random.rand(len(self.layers[3]), 1) ]

        # Network performance metrics
        self.accuracy = 0.0
        self.cost = 0.0

    # Propagates through neural net layers to determine learning step
    # Params: dCda: Derivative of cost function wrt activations for final layer
    # Return: gradient steps for weights and biases
    def backprop(self, dCda):

        # This contains the gradient of the weights and biases that we will return
        # Changes for each step
        weight_gradient = [ np.zeros(shape=(len(self.layers[1]), len(self.layers[0]))),
                            np.zeros(shape=(len(self.layers[2]), len(self.layers[1]))),
                            np.zeros(shape=(len(self.layers[3]), len(self.layers[2]))) ]
        bias_gradient = [ np.zeros(shape=(len(self.layers[1]), 1)), 
                          np.zeros(shape=(len((self.layers[2])), 1)), 
                          np.zeros(shape=(len((self.layers[3])), 1)) ]

        # Propagating backwards through each layer
        for layer in range(len(self.layers)-1, 1):

            # Chain rule components and other helper variables
            dzdw = self.layers[layer-1].T
            dzdb = 1
            dzda = self.weights[layer-1].T
            z = self.weights[layer-1] @ self.layers[layer-1] + self.biases[layer-1]
            dadz = sigmoid_prime(z) # TODO: Indexed properly? no. Check weights transpose?
            error = dCda * dadz

            # Adjust weight and bias gradients
            weight_gradient[layer-1] = error @ dzdw
            bias_gradient[layer-1] = dzdb * error

            # Adjust dCda for previous layer
            dCda = dzda @ error

        return (weight_gradient, bias_gradient) # Should ouput a matrix not a scalar

    # Stochastic gradient descent
    # Params: batch_size: Size of batch for each gradient descent step
    def gradient_descent(self, batch_size):

        for _ in range(batch_size):
            # TODO: Finish this



# -------- Helper Functions (Not part of class) --------

# Cost function and derivative
# MSE between the expected and actual value and derivative of that value
# Params: observed: vector outputted by model
#         expected: scalar value that was expected
# Returns: cost of output values (or derivative)
def cost(observed, expected):
    expected_vector = np.zeros(shape=(10, 1))
    expected_vector[expected] = 1
    return np.sum( (observed - expected_vector)**2 )
def cost_prime(observed, expected):
    expected_vector = np.zeros(shape=(10, 1))
    expected_vector[expected] = 1
    # Note: This is meant to output a vector b/c it depends on what the cost for each activation is 
    return 2 * ( observed - expected_vector )

## The sigmoid activation function and derivative
## Params: z: scalar or vector value
## Returns: sigmoid shifted (element-wise) value between 0 and 1 (or derivative)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

Network(784, 10)