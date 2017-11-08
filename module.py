class module(object):
    """
    An abstract class to keep track of all the parameters
    """
    def __init__(self, **kwargs):
        self.layers = []
        self.name2ix = {}

    def register_layer(self, name, layer):
        self.name2ix[name] = len(self.layers)
        self.layers.append(layer)

    def forward(self, *inputs):
        hidden = None
        for ix in xrange(len(self.layers)):
            hidden = self.layer[ix](*inputs) if hidden is None else self.layer[ix](hidden)
        return hidden

    def backward(self, *inputs):
        hidden = None
        for ix in xrange(len(self.layers) - 1, -1, -1):
            hidden = self.layer[ix].backward(*inputs) if hidden is None else self.layer[ix].backward(hidden)
        return hidden

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = {}
        for name in self.name2ix:
            params[name] = self.layers[self.name2ix[name]].parameters()
        return params
