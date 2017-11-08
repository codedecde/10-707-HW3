'''
Do the experiments
'''
import os
import cPickle as cp
from constants import *
from copy import deepcopy
import heapq as h
import pdb
import numpy as np

model_dir = "models_linear/"
best_val = None
best_file = None
for file in os.listdir(model_dir):
    ppx = float(file.split('_')[-1].strip(".model"))
    if best_val is None or best_val > ppx:
        best_val = ppx
        best_file = model_dir + file
print "LOADING MODEL FROM : %s" % (best_file)
model = cp.load(open(best_file))
vocab = cp.load(open(VOCAB_FILE))


def generate_language(sent, vocab, model):
    """
    Generates a sentence, given the sent
    """
    sent = [vocab[w] if w in vocab else vocab["<UNK>"] for w in sent.split(' ')]
    ix = 0
    ix2vocab = {vocab[w]: w for w in vocab}
    gen_s = deepcopy(sent)
    while ix != 10:
        inp = np.array(sent).reshape(1, -1)
        probs = model(inp)
        # Sample from the model
        sample = np.random.multinomial(100, probs.flatten(), size=1)
        pred = np.argmax(sample)
        sent = sent[1:] + [pred]
        gen_s.append(pred)
        ix += 1
        if ix2vocab[pred] == END_TOK:
            break
    return ' '.join([ix2vocab[jx] for jx in gen_s])

print generate_language("city of new", vocab, model)


def distance(v1, v2):
    return ((v1 - v2) ** 2).sum()


def find_nearest_k(word, vocab, model, k=5):
    ix2vocab = {vocab[w]: w for w in vocab}
    vocab_matrix = model.embedding_layer.parameters()['W'].data
    word_ix = vocab[word]
    word_embed = vocab_matrix[word_ix]
    top_k = []
    for ix in xrange(vocab_matrix.shape[0]):
        if ix == word_ix:
            continue
        dist = -distance(word_embed, vocab_matrix[ix])
        if len(top_k) < k:
            h.heappush(top_k, (dist, ix2vocab[ix]))
        else:
            if top_k[0][0] < dist:
                h.heappop(top_k)
                h.heappush(top_k, (dist, ix2vocab[ix]))
    return top_k
