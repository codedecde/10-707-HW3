from Module import module
import Layers as L
from constants import *


class LanguageModel(module):
    def __init__(self, vocab_size, n_embed, n_hidden, activation):
        super(LanguageModel, self).__init__()
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.embedding_layer = L.EmbeddingLayer(in_dim=vocab_size, h_dim=n_embed)
        self.register_layer("embedding_layer", self.embedding_layer)
        self.hidden_layer = L.DenseLayer(N_GRAM * n_embed, n_hidden, activation=activation)
        self.register_layer("hidden_layer", self.hidden_layer)
        self.output_layer = L.DenseLayer(n_hidden, vocab_size, activation="softmax")
        self.register_layer("output_layer", self.output_layer)

    def forward(self, inp):
        """
        The forward pass.
            :param inp: batch x 3: list of integers
            :return output: batch x vocab_size: The probability distribution
        """
        embed = self.embedding_layer(inp)  # batch x 3 x n_embed
        embed = embed.reshape(embed.shape[0], -1)  # batch x (3 * n_embed)
        encoded_embed = self.hidden_layer(embed)  # batch x n_hidden
        output_embed = self.output_layer(encoded_embed)  # batch x vocab_size
        return output_embed

    def backward(self, out):
        """
        The backward pass
            :param out: batch x vocab_size: The output gradient
            :return gradient: The gradient wrt current node
        """
        _grad = self.output_layer.backward(out)  # batch x n_hidden
        _grad = self.hidden_layer.backward(_grad)  # batch x (3 * n_embed)
        _grad = _grad.reshape(_grad.shape[0], -1, self.n_embed)  # batch x 3 x n_embed
        self.embedding_layer.backward(_grad)  # batch x 3
