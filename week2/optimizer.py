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

alpha is the usual step size parameter
epsilon is some small number that is meant to stabilize the denom to not be 0
"""

import numpy as np


class ADAM:

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
        """

        Args:
            weights (array of dicts): the weight dicts from the network class
            gradient (array of dicts): in our case, this will be the TD-error * gradient

        Returns:
            updated weights dicts
        """

        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * gradient[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * gradient[i][param] ** 2

                m_hat = self.m[i][param] / (1 - self.beta_m_exp)
                v_hat = self.v[i][param] / (1 - self.beta_v_exp)

                weights[i][param] = weights[i][param] - self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat

        # Now we need to take care of our exponentially increasing beta's
        self.beta_m_exp *= self.beta_m
        self.beta_v_exp *= self.beta_v

        return weights


