"""
Contains the NN class, meant to be used as a single-hidden-layer dense neural network.
"""

import numpy as np
from copy import deepcopy


class NN:
    """
    This is meant to be used as a single-hidden-layer neural network.
    
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
        """"""

        self.input_length = input_length
        self.n_hidden_units = n_hidden_units
        self. n_outputs = n_outputs
        self.seed = seed

        self.random_generator = np.random.RandomState(seed)

        self.layer_dims = [self.input_length, self.n_hidden_units, self.n_outputs]
        self.weights = [dict() for i in range(len(self.layer_dims) - 1)]
        for i in range(0, len(self.layer_dims) - 1):
            self.weights[i]['W'] = self._init_saxe(rows=self.layer_dims[i], cols=self.layer_dims[i + 1])
            self.weights[i]['b'] = self._init_saxe(rows=1, cols=self.layer_dims[i + 1])

    def _init_saxe(self, rows, cols):
        """
        Args:
            rows (int): number of input units for layer.
            cols (int): number of output units for layer.
        Returns:
            np.array consisting of weights for the layer based on the initialization in Saxe et al.
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
            the q-values, np.array(a1, a2, a3)
        """
        W_0 = self.weights[0]['W']
        b_0 = self.weights[0]['b']

        phi = np.maximum(0, np.dot(input, W_0) + b_0)

        W_1 = self.weights[1]['W']
        b_1 = self.weights[1]['b']

        # Z_0, Z_1
        if partial:
            # return np.dot(input, W_0) + b_0, np.dot(phi, W_1) + b_1
            return phi

        else:
            return np.dot(phi, W_1) + b_1

    def get_weights(self):
        """
        Returns:
            A copy of the current weights of this network.
        """
        return deepcopy(self.weights)

    def set_weights(self, weights):
        """
        Args:
            weights (list of dictionaries): Consists of weights that this network will set as its own weights.
        """
        self.weights = deepcopy(weights)

    def softmax(self, input):
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

    def softmax_grad(self, y_pred, y_actual):
        """

        Args:
            y_pred:
            y_actual:

        Returns:

        """
        y_pred[range(y_pred.shape[0]), y_actual] -= 1
        return y_pred / y_pred.shape[0]

    def get_gradient(self, input, y_pred, y_actual):
        """"""
        # Pulling out the necessary non-activation-function outputs of each layer
        # Z_0, Z_1 = self.forward_pass(input=input, partial=True)
        relu_output = self.forward_pass(input=input, partial=True)

        # dL_dsoftmax
        dL_dsoftmax = self.softmax_grad(y_pred=y_pred, y_actual=y_actual)

        grad = [dict() for i in range(len(self.weights))]

        # The gradient of our biases in the output layer is simply equal to our loss
        dL_db_1 = dL_dsoftmax

        # For the weights in the 2nd layer, deriv of activation in that layer (ReLU).T @ loss
        dL_dW_1 = np.dot(relu_output.T, dL_dsoftmax)

        # Moving back one layer, let's now do the gradient of the 1st layer's biases which is loss @ W1.T
        dL_db_0 = np.dot(dL_db_1, self.weights[1]['W'].T)

        dL_dW_0 = np.dot(input.T, dL_db_0)

        grad[0]['W'] = dL_dW_0
        grad[0]['b'] = dL_db_0
        grad[1]['W'] = dL_dW_1
        grad[1]['b'] = dL_db_1

        return grad


    # #
    # def get_gradient(self, input, error):
    #     """"""
    #     grads = [dict() for i in range(len(self.weights))]
    #
    #     W_0, b_0 = self.weights[0]['W'], self.weights[0]['b']
    #     W1, b1 = self.weights[1]['W'], self.weights[1]['b']
    #
    #     phi = np.maximum(0, np.dot(input, W_0) + b_0)
    #
    #     dv_db_1 = 1
    #     dv_dw_1 = phi.T
    #     print(phi)
    #     I = np.ones((phi.shape[0], phi.shape[1]))
    #     I[phi <= 0] = 0
    #     print(I)
    #     print(self.weights[1]['W'])
    #     dv_db_0 = self.weights[1]['W'].T * I
    #     print(dv_db_0)
    #     assert(1 == 0)
    #
    #     delta_v_delta_b1 = 1
    #     delta_v_delta_W1 = phi.T
    #
    #
    #     # Gradient of the ReLU activation function
    #     I = np.ones((delta_v_delta_W1.shape[0], delta_v_delta_W1.shape[1]))
    #     I[delta_v_delta_W1 < 0] = 0
    #
    #     delta_v_delta_b0 = self.weights[1]['W'].T * I
    #     delta_v_delta_W0 = np.dot(input.T, delta_v_delta_b0)
    #
    #     grads[0]['W'] = delta_v_delta_W0 * error
    #     grads[0]['b'] = delta_v_delta_b0 * error
    #     grads[1]['b'] = delta_v_delta_b1 * error
    #     grads[1]['W'] = delta_v_delta_W1 * error
    #
    #     return grads
    #
    # # def get_gradient(self, forward_pass_output, y_actual):
    # #     """"""
    # #
    # #     # Initial expected values...
    # #     expected = np.zeros((y_actual.shape[0], forward_pass_output.shape[1]))
    # #     expected[range(y_actual.shape[0], y_actual)] = 1
    # #
    # #     # Working backwards from the end of the network...
    # #     # (0) Softmax
    # #     softmax_gradient = forward_pass_output[range(y_actual.shape[0]), y_actual]
    # #     softmax_gradient -= 1
    # #     softmax_gradient = softmax_gradient / y_actual.shape[0]
    # #
    # #     error = (expected - forward_pass_output) * softmax_gradient
    #
    #






