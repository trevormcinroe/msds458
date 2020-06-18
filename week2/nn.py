"""
Contains the NN class, meant to be used as a single-hidden-layer dense neural network.
"""

import numpy as np


class NN:
    """
    This is meant to be used as a single-hidden-layer neural network.
    Uses a ReLU activation function after the first layer and a softmax output.
    
    Attributes:
        User inputted:
            input_length (int): the dimension of the input vector
            n_hidden_units (int): the number of units in the hidden layer
            n_outputs (int): in the XOR case, the number of classes, usually 2
            seed (int): the random seed to allow for experiment reproducibility

        Class managed:
            random_generator (RandomState): random number generator that inherits the user-inputted seed
            layer_dims (list): the input/output size of each layer, used to build the layers
            weights (list of dicts): the actual learnable weights of the network

    """
    def __init__(self, input_length, n_hidden_units, n_outputs, seed):

        self.input_length = input_length
        self.n_hidden_units = n_hidden_units
        self. n_outputs = n_outputs
        self.seed = seed

        self.random_generator = np.random.RandomState(seed)

        self.layer_dims = [self.input_length, self.n_hidden_units, self.n_outputs]
        self.weights = [dict() for _ in range(len(self.layer_dims) - 1)]
        for i in range(0, len(self.layer_dims) - 1):
            self.weights[i]['W'] = self._init_saxe(rows=self.layer_dims[i], cols=self.layer_dims[i + 1])
            self.weights[i]['b'] = self._init_saxe(rows=1, cols=self.layer_dims[i + 1])

    def _init_saxe(self, rows, cols):
        """Saxe weight initialization as based on Saxe et al. (2013) https://arxiv.org/pdf/1312.6120.pdf

        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            np.array of initialized weights
        """
        tensor = self.random_generator.normal(0, 1, (rows, cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T
        return tensor

    def forward_pass(self, input, partial=False):
        """Computes a forward pass through the network.

        Args:
            input (np.array): input data
            partial (bool): used to pull out relevant information during the gradient calculation step

        Returns:
            np.array of predicted values [batch_size, n_outputs]
        """
        W_0 = self.weights[0]['W']
        b_0 = self.weights[0]['b']

        phi = np.maximum(0, np.dot(input, W_0) + b_0)

        W_1 = self.weights[1]['W']
        b_1 = self.weights[1]['b']

        # During gradient calculations, we only need the activated-output of the first layer
        if partial:
            return phi

        else:
            return np.dot(phi, W_1) + b_1

    @staticmethod
    def softmax(input):
        """Computes the softmax transformation from the output of the forward pass of the network. This is the version
        of softmax that is numerically stable. That is, it is not prone to either underflow or overflow.

        Args:
            input (np.array): 2d array [batch_size, n_outputs]
        Returns:

        """
        max_stablizer = np.max(input, axis=1).reshape((-1, 1))
        numerator = np.exp(input - max_stablizer)
        denominator = np.sum(numerator, axis=1).reshape((-1, 1))
        return (numerator / denominator).squeeze()

    @staticmethod
    def softmax_grad(y_pred, y_actual):
        """Handles the computation of the derivative of our softmax output.

        Args:
            y_pred (np.array, 2D): [batch_size, n_outputs]
            y_actual (np.array, 1D): [batch_size,]

        Returns:
            np.array, 2D - the gradient
        """
        y_pred[range(y_pred.shape[0]), y_actual] -= 1
        return y_pred / y_pred.shape[0]

    def get_gradient(self, input, y_pred, y_actual):
        """Look at that, the network can compute its own gradients!

        Args:
            input (np.array): our training examples [batch_size, n_outputs]
            y_pred (np.array): the predicted values (POST SOFTMAX) [batch_size, n_outputs)
            y_actual (np.array): the ground-truth labels [batch_size,]

        Returns:
            a list of dictionaries with the same structure as the self.weights attribute
        """
        # For an explanation of what is going on, please see the "Derivation of the backprop algorithm" section of
        # the README.
        relu_output = self.forward_pass(input=input, partial=True)

        dL_dsoftmax = self.softmax_grad(y_pred=y_pred, y_actual=y_actual)

        grad = [dict() for i in range(len(self.weights))]

        dL_db_1 = dL_dsoftmax

        dL_dW_1 = np.dot(relu_output.T, dL_dsoftmax)

        dL_db_0 = np.dot(dL_db_1, self.weights[1]['W'].T)

        dL_dW_0 = np.dot(input.T, dL_db_0)

        grad[0]['W'] = dL_dW_0
        grad[0]['b'] = dL_db_0
        grad[1]['W'] = dL_dW_1
        grad[1]['b'] = dL_db_1

        return grad
