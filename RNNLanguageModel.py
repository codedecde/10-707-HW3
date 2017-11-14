import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_hidden, activation='tanh'):
        super(RNNLanguageModel, self).__init__()
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, n_embed)
        self.rnn = nn.RNN(n_embed, n_hidden, nonlinearity='tanh', num_layers=1)
        self.output = nn.Linear(n_hidden, vocab_size)

    def init_hidden(self, batch):
        return Variable(torch.zeros((1, batch, self.n_hidden)))

    def forward(self, inp, truncate=None):
        """The forward pass
            :param inp: batch x 3: List of integers
            :param truncate: int: truncate backpropagation after truncate steps.
                                : Default None
            :return output: batch x vocab : Logsoftmax of the words
        """
        embed = self.embedding_layer(inp)  # batch x T x n_embed
        embed = embed.transpose(0, 1)  # T x batch x n_embed
        h0 = self.init_hidden(embed.size(1))
        if truncate is None:
            o, _ = self.rnn(embed, h0)  # o : T x batch x n_hidden, # h : 1 x batch x n_hidden
        else:
            # pdb.set_trace()
            _, h = self.rnn(embed[:embed.size(0) - truncate], h0)
            h.detach_()  # to detatch from computational graph
            o, _ = self.rnn(embed[embed.size(0) - truncate:], h)

        return F.log_softmax(self.output(o[-1]))  # batch x vocab_size
