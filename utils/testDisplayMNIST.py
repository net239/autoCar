# Tested with Python 3.5.2 with tensorflow and matplotlib installed.
from matplotlib import pyplot as plt
import numpy as np

# import handwritten images
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# just a small code to display any random image in the database
upperbound = 256
lowerbound = 0
idx = (upperbound-lowerbound)*np.random.random((1, 1)) + lowerbound
idx = int(idx)
print idx


# display the image
first_image = mnist.test.images[idx]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.title(mnist.test.labels[idx])
plt.show()
