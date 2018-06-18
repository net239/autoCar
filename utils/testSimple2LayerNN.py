import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# sigmoid activation function and its derivative
def nonlin(x, deriv=False):
    if (deriv is True):
        return x*(1-x)

    return 1 / (1 + np.exp(-x))


# define the input and output vector
# input is total 4 samples, with 3 dimensions each
X = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 0]
])

# we define  output as mirror of 3rd column
Y = np.array([
    [1],
    [0],
    [1],
    [0]
])

# number fo input features and number of samples
n_samples, n_features = X.shape
ignore, n_predictions = Y.shape

# initialize random seed
np.random.seed(1)

# lets define two weight matrices for this 2 layer fully connected network
# the first weight matrix connects input X (layer0) to layer1
# and the second weight matrix connects inout from layer0 to layer2
# the weights are assigned randomly to start with and are between -1 and 1
# with mean as zero
upperbound = 1
lowerbound = -1
syn0 = (upperbound-lowerbound)*np.random.random((n_features, n_samples)) + \
                                                 lowerbound
syn1 = (upperbound-lowerbound)*np.random.random((n_samples, n_predictions)) + \
                                                 lowerbound

# use these to see progress of all datapoints int the network
# use to store all errors
track_layer1 = pd.DataFrame()
track_layer2 = pd.DataFrame()
track_error1 = pd.DataFrame()
track_error2 = pd.DataFrame()

print "n_features ", n_features
print "n_samples ", n_samples
print "n_predictions ", n_predictions

# lets train our network
for j in xrange(10000):
    # Lets first apply the weights to the input and forward to layer1
    # and then to layer2
    layer0 = X
    layer1 = nonlin(np.dot(layer0, syn0))
    layer2 = nonlin(np.dot(layer1, syn1))

    # lets see how far away we are from the actual solution
    # start from the last layer and move backward
    error2 = Y - layer2

    # see how much we need to adjust weights
    # error times slope
    delta2 = error2 * nonlin(layer2, deriv=True)

    # how much did each l1 value contribute to the l2 error
    # (according to the weights)?
    error1 = np.dot(delta2, syn1.T)

    # see how much we need to adjust weights
    # error times slope
    delta1 = error1 * nonlin(layer1, deriv=True)

    syn1 += np.dot(layer1.T, delta2)
    syn0 += np.dot(layer0.T, delta1)

    # store all data, so we can analyze them later
    data = layer1.T
    track_layer1 = track_layer1.append(pd.DataFrame(data), ignore_index=True)

    data = syn0.T
    track_layer2 = track_layer2.append(pd.DataFrame(data), ignore_index=True)

    data = error1.T
    track_error1 = track_error1.append(pd.DataFrame(data), ignore_index=True)

    data = error2.T
    track_error2 = track_error2.append(pd.DataFrame(data), ignore_index=True)

# plot the data generated in above iterations
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
axs[0, 0].plot(track_layer1, label="layer1")
axs[0, 0].set_title("layer1")

axs[0, 1].plot(track_error1, label="error1")
axs[0, 1].set_title("error1")

axs[0, 2].plot(track_layer2, label="layer2")
axs[0, 2].set_title("layer2")

axs[1, 1].plot(track_error2, label="error2")
axs[1, 1].set_title("error2")

print layer2
plt.show()
