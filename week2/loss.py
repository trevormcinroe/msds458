"""

"""

import numpy as np


def cross_entropy_loss(y_pred, y_actual):
    """

    Args:
        y_pred: the output of the network (in post-softmax form)
        y_actual: NOT one-hot encoded

    Returns:

    """
    log_likelihood = -np.log(y_pred[range(y_actual.shape[0]), y_actual])
    return np.sum(log_likelihood) / y_actual.shape[0]
