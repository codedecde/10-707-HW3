'''
The language model, but in pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import cPickle as cp
from constants import *
import pdb
from utils import Progbar
import numpy as np


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_hidden, activation):
        super(LanguageModel, self).__init__()
        self.n_embed = n_embed
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.activation = activation
        self.embedding_layer = nn.Embedding(self.vocab_size, self.n_embed)
        self.hidden_layer = nn.Linear(N_GRAM * n_embed, n_hidden)
        self.output_layer = nn.Linear(n_hidden, vocab_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, inp):
        """
        The forward pass.
            :param inp: batch x 3: list of integers
            :return output: batch x vocab_size: The probability distribution
        """
        embed = self.embedding_layer(inp)  # batch x 3 x n_embed
        embed = embed.view(embed.size(0), -1)  # batch x (3 * n_embed)
        encoded_embed = self.hidden_layer(embed)  # batch x n_hidden
        if self.activation != "linear":
            encoded_embed = getattr(F, self.activation)(encoded_embed)
        output_embed = self.log_softmax(self.output_layer(encoded_embed))  # batch x vocab_size
        return output_embed


# ========== Load the Vocab ==============#
vocab = cp.load(open("data/vocab.pkl"))
# ========================================#


# ========== Data Indexing ===============#
def index_data(data, vocab):
    ret_data = []
    for ix in xrange(len(data)):
        data[ix] = data[ix].split(' ')
        data[ix] = [START_TOK] + data[ix] + [END_TOK]
        for jx in xrange(len(data[ix]) - N_GRAM):
            ngram = [vocab[w] if w in vocab else vocab["<UNK>"] for w in data[ix][jx: jx + N_GRAM]]
            pred = vocab[data[ix][jx + N_GRAM]] if data[ix][jx + N_GRAM] in vocab else vocab["<UNK>"]
            ret_data.append((ngram, pred))
    return ret_data


# ========== Define Constants ============#
NUM_EMBED = 16
NUM_HIDDEN = 128
ACTIVATION = "tanh"
N_EPOCHS = 100
BATCH_SIZE = 128
# ========================================#
# ========== Training Loop ===============#
lm = LanguageModel(len(vocab), NUM_EMBED, NUM_HIDDEN, ACTIVATION)
training_data = map(lambda x: x.lower().strip(), open("data/train.txt").readlines())
indexed_training_data = index_data(training_data, vocab)
loss = nn.NLLLoss()
optimizer = optim.SGD(lm.parameters(), lr=0.01)
for epoch in xrange(N_EPOCHS):
    print "\nEPOCH ({}/{})".format(epoch + 1, N_EPOCHS)
    steps = -(-len(indexed_training_data) // BATCH_SIZE)  # Round up
    bar = Progbar(steps)
    for step, ix in enumerate(xrange(0, len(indexed_training_data), BATCH_SIZE)):
        batch_x, batch_y = map(lambda x: np.array(x, dtype=int), zip(*indexed_training_data[ix: ix + BATCH_SIZE]))
        batch_x = autograd.Variable(torch.LongTensor(batch_x))
        preds = lm(batch_x)
        batch_y = autograd.Variable(torch.LongTensor(batch_y))
        _l = loss(preds, batch_y)
        bar.update(step + 1, values=[("train_loss", _l.data.numpy()[0])])
        _l.backward()
        optimizer.step()
        optimizer.zero_grad()
    # TODO: Validation stuff
# ======== End of Training Loop ===========#
