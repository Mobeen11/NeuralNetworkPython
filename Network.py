import numpy as np
import random


"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class Network(object):

    def __init__(self, size):

        """
        Initialize the Network
        num_layers: Number of Layers
        size: Number of Neurons in Layer  e.g: num = Network([784,30,10])   2 Neuron in Layer 1, 3 Neuron in Layer 2 and 1 Neuron in Output Layer
        biase: Biase Bit to increase the training speed
        weights: Weights of the Neuron
        """

        # constructor for Initialize the network
        self.num_layers = len(size)
        self.size = size
        self.biase = [np.random.randn(y, 1) for y in size[1:]]          #  np.random.randn for the Gaussian Distribution Mean:0 & Standard Deviation: 1
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(size[:-1], size[1:])]

    def feedForward(self, a):
        """
            Forward Propagation return the output of Network
        :param a:
        :return: it returns the product of sums of all neurons
        """
        for b, w in zip(self.biase, self.weights):
            a = sigmoid(np.dot(w, a)+b)             # np.dot(w, a) for the Matrix Mutliplication
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        SGD Means Stochastic Gradient Descent
        Function for the Training the Network

            Traning data is in the List of Tuples (x,y)
            epochs: One Complete Cycle of Forward Propagation / Backward Propagation on training data
            mini_batch_size: the part of data on which the Training is applied
            eta: rate on which the weights are updated on each term
        :return:
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
                # print "Epoch {0}: {1} / {2}".format(
                #     j, self.evaluate(training_data), n)
            else:
                print "Epoch {0} complete".format(j)
                print ('No Test Data')
    """
        The Network works in the way it divide the data in mini_batch and apply Training on that batch
    """

    def update_mini_batch(self, mini_batch, eta):
        """
            Update the network's weights and biase by applying
            gradient descent using backpropagation
        """
        nabla_b = [np.zeros(b.shape) for b in self.biase]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)              # this CAlls the Backword Propagation
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biase = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biase, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biase`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biase]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedForward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biase, self.weights):
            z = np.dot(w, activation)+b             # activation Function
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. argmax function will give the probability of the ouptut """
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return: 1/1+e^-x
    """
    return 1.0/(1.0+np.exp(-x))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

training_data, validation_data, test_data = load_data_wrapper()
net = Network([784, 30, 10])


net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
