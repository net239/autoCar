import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              ])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(6)

# initialize weights randomly with mean 0
synapse_0 = 2*np.random.random((2, 1)) - 1

# use to store all errors
track_errors = pd.DataFrame()
track_activf = pd.DataFrame()
track_dweights = pd.DataFrame()
track_dlayer1 = pd.DataFrame()
track_layer1 = pd.DataFrame()
track_synapse_0 = pd.DataFrame()

for iter in xrange(100):

    # forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))

    # how much did we miss?
    layer_1_error = layer_1 - y

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0_derivative = np.dot(layer_0.T, layer_1_delta)

    # store all slopes, so we can analyze them later
    data = layer_1.T
    track_layer1 = track_layer1.append(pd.DataFrame(data), ignore_index=True)

    data = layer_1_error.T
    track_errors = track_errors.append(pd.DataFrame(data), ignore_index=True)

    data = sigmoid_output_to_derivative(layer_1).T
    track_activf = track_activf.append(pd.DataFrame(data), ignore_index=True)

    data = layer_1_delta.T
    track_dlayer1 = \
        track_dlayer1.append(pd.DataFrame(data), ignore_index=True)

    data = synapse_0_derivative.T
    track_dweights = \
        track_dweights.append(pd.DataFrame(data), ignore_index=True)

    # update weights
    synapse_0_derivative = synapse_0_derivative * 10
    synapse_0 -= synapse_0_derivative
    data = synapse_0.T
    track_synapse_0 = \
        track_synapse_0.append(pd.DataFrame(data), ignore_index=True)

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].plot(track_errors, label="errors")
axs[0, 0].set_title("errors")

axs[0, 1].plot(track_layer1, label="layer1")
axs[0, 1].set_title("layer1")

axs[0, 2].plot(track_activf, label="activation function")
axs[0, 2].set_title("activf")

axs[1, 0].plot(track_dweights, label="delta weights")
axs[1, 0].set_title("dweights")

axs[1, 1].plot(track_dlayer1, label="delta layer1")
axs[1, 1].set_title("dlayer1")

axs[1, 2].plot(track_synapse_0, label="weights")
axs[1, 2].set_title("weights")


print "Output After Training:"
print layer_1
plt.show()
