



### TODO: DESIGN THIS PROJECT ON PAPER/WHITEBOARD BEFORE DOING MORE






import math
import numpy as np
import pandas as pd
from pprint import pprint

import matplotlib_terminal
import matplotlib.pyplot as plt


train = pd.read_csv("data/train.csv")
y = train["label"]
data = train.drop(["label"], axis=1).to_numpy()

print(train.shape)

#pprint(y.head())
#pprint(data.head())
#print(data.shape[1])

# Number of layers including input and output layers
# input layer is = to number of columns in dataframe
layers = [ np.zeros(shape=(data.shape[1], 1)), np.zeros(shape=(16,1)), np.zeros(shape=(16,1)), np.zeros(shape=(10,1)) ]

# TODO: Do i need a separate thing here then?
biases = [ np.random.rand(len(layers[1]), 1), 
           np.random.rand(len(layers[2]), 1), 
           np.random.rand(len(layers[3]), 1) ]

#print(layers[1])

# array of weights between each of the 4 layers (2 hidden layers), length is n-1
# w_jk is how we index these
# We start with random values since that seems to work better than 0s or 1s
weights = [ np.random.rand(len(layers[1]), len(layers[0])), 
            np.random.rand(len(layers[2]), len(layers[1])), 
            np.random.rand(len(layers[3]), len(layers[2])) ]



weight_gradient = [ np.zeros(shape=(len(layers[1]), len(layers[0]))),
                    np.zeros(shape=(len(layers[2]), len(layers[1]))),
                    np.zeros(shape=(len(layers[3]), len(layers[2]))) ]
bias_gradient = [ np.zeros(shape=(len(layers[1]), 1)), np.zeros(shape=(len((layers[2])), 1)), np.zeros(shape=(len((layers[3])), 1)) ]



# Cost function is the MSE between the expected and actual value
# Params: observed: vector outputted by model
#         expected: scalar value that was expected
# Returns: cost of output values
def cost(observed, expected):
    expected_vector = np.zeros(shape=(10, 1))
    expected_vector[expected] = 1
    return np.sum( (observed - expected_vector)**2 )

def cost_prime(observed, expected):
    expected_vector = np.zeros(shape=(10, 1))
    expected_vector[expected] = 1
    # Note: This is meant to output a vector b/c it depends on what the cost for each activation is TODO: Or is it?
    return 2 * ( observed - expected_vector )

## The sigmoid activation function and derivative
## Params: z: scalar or vector value
## Returns: sigmoid shifted (element-wise) value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

# ## Neuron value given inputs, weights, and biases
# ## Params: inputs: vector of input values
# ##         weights: matrix of weights from input values
# ##         bias: scalar/vector for threshold shift
# ## Returns: value between 0 and 1
# def neuron(inputs, weights, bias):
#     # product of inputs and weights plus bias
#     return sigmoid(inputs @ weights + bias) # TODO: maybe remove this function

# DOC THIS TODO -- layer to be processes
# assumes previous layers are processed already
# maybe make this a process all layers function
def process_layer(layer):
    # print(weights[layer-1].shape)
    # print(layers[layer-1].shape)
    # print(biases[layer-1].shape)
    layers[layer] = sigmoid(weights[layer-1] @ layers[layer-1] + biases[layer-1]) # TODO: How do i handle biases
#print(neuron(np.array([[3,-4,5]]), np.array([[2,1,-0.2], [1,2,-0.4]]).T, [1,2]))


# Returns gradient steps for each weight and biases
def backprop(layer, dCda):
    #print("backprop working")

    if(layer <= 0): return

    dzdw = layers[layer-1].T
    dzdb = 1
    dzda = weights[layer-1].T # weights from input to every output -- TODO: is this transposed?

    z = weights[layer-1] @ layers[layer-1] + biases[layer-1]

    dadz = sigmoid_prime(z) # TODO: Indexed properly? no. Check weights transpose?
    error = dCda * dadz

    # print("backprop")
    # pprint(dzdw.shape)
    # pprint(dzda.shape)
    # pprint(dadz.shape)
    # pprint(error.shape)
    # print("/backprop")

    weight_gradient[layer-1] = error @ dzdw
    bias_gradient[layer-1] = dzdb * error
    # weight_gradient[layer-1] = dzdw @ (dCda * dadz)
    # bias_gradient[layer-1] = dzdb * (dadz * dCda)

    return backprop(layer - 1, dzda @ error) # Should ouput a matrix not a scalar


def gradient_descent(batch_size):
    print("descending gradient")
    
    avg_weight_gradient = [ np.zeros(shape=(len(layers[1]), len(layers[0]))), 
                            np.zeros(shape=(len(layers[2]), len(layers[1]))), 
                            np.zeros(shape=(len(layers[3]), len(layers[2]))) ]
    # TODO: Avg bias gradient uses newaxis but others don't??
    avg_bias_gradient = [ np.zeros(shape=(len(layers[1]), 1)), np.zeros(shape=(len((layers[2])), 1)), np.zeros(shape=(len((layers[3])), 1)) ]
    avg_cost = 0.0
    correct = 0

    training_example = 0 # FIXME: Remove this

    for _ in range(batch_size):
        # Select random number in total vals
        training_example = np.random.randint(train.shape[0])

        # TODO: Random sampling w/ replacement
        layers[0] = data[training_example][np.newaxis].T / 255
        for i in range(len(layers)-1): process_layer(i+1)

        avg_cost += cost(layers[3], y[training_example])
        correct += int(layers[3].argmax() == y[training_example])

        # TODO: Remove this?
        weight_gradient = [ np.zeros(shape=(len(layers[1]), len(layers[0]))),
                            np.zeros(shape=(len(layers[2]), len(layers[1]))),
                            np.zeros(shape=(len(layers[3]), len(layers[2]))) ]
        bias_gradient = [ np.zeros(shape=(len(layers[1]), 1)), np.zeros(shape=(len((layers[2])), 1)), np.zeros(shape=(len((layers[3])), 1)) ]

        last_layer = 3
        backprop(last_layer, cost_prime(layers[last_layer], y[training_example]))

        # Update gradients for weights and biases
        for i in range(len(biases)):
            # pprint(avg_bias_gradient)
            # pprint(bias_gradient)
            avg_weight_gradient[i] += weight_gradient[i]
            avg_bias_gradient[i] += bias_gradient[i] # TODO: Should this be transposed?
    

    pprint(y[training_example])
    pprint(layers[3])
    pprint(cost(layers[3], y[training_example]))

    # Divide all gradients and costs by batch size
    avg_weight_gradient = [x / batch_size for x in avg_weight_gradient]
    avg_bias_gradient = [x / batch_size for x in avg_bias_gradient]
    avg_cost = avg_cost/batch_size
    accuracy = correct/batch_size
    
    return (avg_weight_gradient, avg_bias_gradient, avg_cost, accuracy)


def learn(step, batch_size, epochs):
    print("learning lol")

    # TODO: Add random sampling here
    for _ in range(epochs * int(train.shape[0] / batch_size) ):
    # for _ in range(1):
        # Calculate average gradient steps
        grad = gradient_descent(batch_size)
        for j in range(len(biases)):
            weights[j] -= step * grad[0][j]
            biases[j] -= step * grad[1][j]
        print("Cost: " + str(grad[2]))
        print("Accuracy: " + str(grad[3]))
        # print(grad[0])






# #print("Backpropagating:")
# # gradient_step = backprop(3, 2 * cost_prime(layers[3], y[0]) )
# backprop(3, cost_prime(layers[3], y[0]) )
#pprint(weight_gradient)
#pprint(bias_gradient)


learn(1, 10000, 5)
# learn(1, 1, 1)
out = []
#This is how to run a sample set with the neural network
for i in range(2000, 6000):
    layers[0] = data[i][np.newaxis].T / 255 # Set first layer as normalized values
    process_layer(1)
    process_layer(2)
    process_layer(3)
    print("Output (layers, and cost): ")
    out.append(layers[3].argmax())
    pprint(layers[3].argmax())
    print(y[i])
    print(cost(layers[3], y[i]))
    
    # pprint(weights)

plt.hist(out)
plt.show('gamma')
plt.close()


#pprint(layers[3])
#pprint(cost(layers[3], y[2000]))

# pprint(grad)
