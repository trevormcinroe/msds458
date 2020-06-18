"""
The Adam algorithm improves the SGD update with two concepts: adaptive vector stepsizes and momentum.
It keeps running estimates of the mean and second moment of the updates, denoted by:

m_t = B_m * m_t-1 + (1 - B_m)g1_t
v_t = B_v * v_t-1 + (1 - B_v)g^2_t

B_m, B_v are fixed parameters
The g's are the gradients
Given that m, v are init to 0, they are biased towards 0. To get unbiased estimates,
we use...
\hat{m_t} = m_t / (1 - B^t_m)
\hat{v_t} = v_t / (1 - B^t_v)

NOTE: in the above calculations, the beta parameters are raised to the power of the timestep

Using the above, the weight updates are ultimately...
w_t = w_t-1 + (alpha / sqrt(\hat{v_t} + epsilon)) * \hat{m_t}
"""

import numpy as np


class ADAM:
    """
    This is a bare-bones implementation of the ADAM optimizer based on Kingma, Ba (2014) https://arxiv.org/abs/1412.6980

    Attributes:
        layer_dims (list): should be directly passed from the NN.layer_dims attribute to ensure compatibility
        alpha (float): the step-size (learning rate)
        beta_m (float): the coefficient for the "momentum" hyperparameter
        beta_v (float): the coefficient for the "velocity" hyperparameter
        epsilon (float): a very small value, meant to be a stabilizer for when v_hat may be 0
    """
    def __init__(self, layer_dims, alpha, beta_m, beta_v, epsilon):
        self.layer_dims = layer_dims
        self.alpha = alpha
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.epsilon = epsilon

        self.m = [dict() for _ in range(len(self.layer_dims) - 1)]
        self.v = [dict() for _ in range(len(self.layer_dims) - 1)]

        for i in range(len(self.layer_dims) - 1):
            self.m[i]['W'] = np.zeros((self.layer_dims[i], self.layer_dims[i + 1]))
            self.m[i]['b'] = np.zeros((1, self.layer_dims[i + 1]))
            self.v[i]['W'] = np.zeros((self.layer_dims[i], self.layer_dims[i + 1]))
            self.v[i]['b'] = np.zeros((1, self.layer_dims[i + 1]))

        # Instead of requiring a global tracker of the step, we can keep track of the product
        self.beta_m_exp = self.beta_m
        self.beta_v_exp = self.beta_v

    def update_weights(self, weights, gradient):
        """The completion of the backpropagation algorithm, applying both the momentum and the gradient to the weights.
        In addition, this method handles the updating of all hyperparameters.

        Args:
            weights (array of dicts): the weight dicts from the network class, use a DIRECT REFERENCE! NOT A COPY!
            gradient (array of dicts): the array of dicts returned by the NN.get_gradient() method

        Returns:
            None. the network's weights are mutable, so this method will actually be mutating the weights of the
            network directly, assuming a direct reference to them was passed to the "weights" argument
        """

        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * gradient[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * gradient[i][param] ** 2

                m_hat = self.m[i][param] / (1 - self.beta_m_exp)
                v_hat = self.v[i][param] / (1 - self.beta_v_exp)

                weights[i][param] = weights[i][param] - self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat

        self.beta_m_exp *= self.beta_m
        self.beta_v_exp *= self.beta_v
