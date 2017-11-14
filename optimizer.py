import numpy as np


class SGD(object):
    def __init__(self, params, lr, momentum=0., l2=0.):
        self.params = params
        self.lr = lr
        self.gamma = momentum
        self.l2 = l2
        self.v = {}
        for name in self.params:
            self.v[name] = {}
            for param in self.params[name]:
                self.v[name][param] = np.zeros(self.params[name][param].grad.shape)

    def step(self):
        for name in self.params:
            for param in self.params[name]:
                self.v[name][param] *= self.gamma
                self.v[name][param] += self.lr * self.params[name][param].grad
                if param != 'b':
                    self.v[name][param] += self.l2 * self.lr * self.params[name][param].data
                self.params[name][param].data -= self.v[name][param]

    def zero_grad(self):
        for name in self.params:
            for param in self.params[name]:
                self.params[name][param].grad = np.zeros(self.params[name][param].data.shape).astype(self.params[name][param].grad.dtype)
