import numpy as np
import pdb


class Loss(object):
    """
    Abstract Base class for all lossses
    """
    def __init__(self):
        self.eps = np.finfo(float).eps  # For numerical stability

    def __call__(self):
        raise NotImplementedError


class categorical_cross_entropy(Loss):
    def __init__(self):
        super(categorical_cross_entropy, self).__init__()

    def __call__(self, y, output, normalize=True):
        output += self.eps
        batch_size = output.shape[0]
        loss = (-1 * np.log(output))[range(batch_size), y].sum()
        if normalize:
            loss /= batch_size
        output[range(batch_size), y] -= 1.
        _grad = output
        return loss, _grad


class binary_cross_entropy(Loss):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def __call__(self, y, output):
        output += self.eps
        batch_size = y.shape[0]
        # units = y.shape[1]
        units = 1.
        loss = -1. * ((y * np.log(output)) + ((1. - y) * (np.log(1. - output))))
        _grad = (output - y) / (output * (1. - output) * units)
        loss = np.sum(loss) / (batch_size * units)
        return loss, _grad


class mean_square_error(Loss):
    def __init__(self):
        super(mean_square_error, self).__init__()

    def __call__(self, y, output):
        output += self.eps
        batch_size = y.shape[0]
        loss = 0.5 * (output - y) * (output - y)
        _grad = (output - y)
        loss = np.sum(loss) / batch_size
        return loss, _grad
