'''
The language model
'''

from Loss import categorical_cross_entropy
from optimizer import SGD
import cPickle as cp
from constants import *
from summary import Summary, History
import pdb
from utils import Progbar, make_directory
import numpy as np
from LanguageModel import LanguageModel
import argparse


# ========== Load the Vocab ==============#
vocab = cp.load(open(VOCAB_FILE))


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


# ========== Computing Perplexity ========#
def index_for_perplexity(data, vocab):
    """
    This indexes for perplexity. The difference is it returns a list of batch objects, one for each sentence
        :param data: [text]: data
        :param vocab: word -> ix
        :return out: the batched data, one for each sentence
    """
    out = []
    for ix in xrange(len(data)):
        data[ix] = [START_TOK] + data[ix].split(' ') + [END_TOK]
        batch = []
        for jx in xrange(0, len(data[ix]) - N_GRAM):
            ngram = [vocab[w] if w in vocab else vocab["<UNK>"] for w in data[ix][jx: jx + N_GRAM]]
            pred = vocab[data[ix][jx + N_GRAM]] if data[ix][jx + N_GRAM] in vocab else vocab["<UNK>"]
            batch.append((ngram, pred))
        out.append(batch)
    return out


def perplexity(idx_for_ppx, model):
    """
    Compute the perplexity of the validation data
        :param idx_for_ppx: Indexed data for perplexity computation
        :param model: The model to evaluate
        :return ppx: The perplexity
    """
    ppx = 0.
    for ix in xrange(len(idx_for_ppx)):
        batch = idx_for_ppx[ix]
        batch_x, batch_y = map(lambda x: np.array(x), zip(*batch))
        preds = model(batch_x)
        lp = np.log2(preds[range(preds.shape[0]), batch_y]).sum()
        ppx += (lp / batch_y.shape[0])
    return 2 ** (-ppx / len(idx_for_ppx))


# ========== Argument Parser ============#
def parse_args():
    parser = argparse.ArgumentParser(description="Language Model")
    parser.add_argument("-ne", "--embed_size", help="Embedding Size", dest="n_embed", default=16, type=int)
    parser.add_argument("-nd", "--hidden_size", help="Hidden Size", dest="n_dim", default=128, type=int)
    parser.add_argument("-a", "--activation", help="Activation", dest="activation", default="linear", type=str)
    parser.add_argument("-epochs", "--num_epochs", help="Number of epochs", dest="n_epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", help="Batch Size", dest="batch", default=128, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning Rate", dest="lr", default=0.1, type=float)
    parser.add_argument("-p", "--patience", help="Patience", dest="patience", default=2, type=int)
    args = parser.parse_args()
    return args


args = parse_args()
NUM_EMBED = args.n_embed
NUM_HIDDEN = args.n_dim
ACTIVATION = args.activation
N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch
LR = args.lr
PATIENCE = args.patience
# ========== Initialize Summary ==========#
meta = {"NUM_EMBED": NUM_EMBED,
        "NUM_HIDDEN": NUM_HIDDEN,
        "ACTIVATION": ACTIVATION,
        "N_EPOCHS": N_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR}
for elem in meta:
    print "{}:\t{}".format(elem, meta[elem])
s = Summary(meta)
best_val = None
# ========== Training Loop ===============#
lm = LanguageModel(len(vocab), NUM_EMBED, NUM_HIDDEN, ACTIVATION)
training_data = map(lambda x: x.lower().strip(), open(TRAIN_FILE).readlines())
val_data = map(lambda x: x.lower().strip(), open(VAL_FILE).readlines())
indexed_training_data = index_data(training_data, vocab)
idx_for_ppx = index_for_perplexity(val_data, vocab)
loss = categorical_cross_entropy()
optimizer = SGD(lm.parameters(), lr=LR, momentum=0.9)
patience_count = 0
for epoch in xrange(N_EPOCHS):
    print "\nEPOCH ({}/{})".format(epoch + 1, N_EPOCHS)
    steps = -(-len(indexed_training_data) // BATCH_SIZE)  # Round up
    bar = Progbar(steps)
    tl = 0.
    for step, ix in enumerate(xrange(0, len(indexed_training_data), BATCH_SIZE)):
        batch_x, batch_y = map(lambda x: np.array(x), zip(*indexed_training_data[ix: ix + BATCH_SIZE]))
        batch_x = np.array(batch_x)
        preds = lm(batch_x)
        l, g = loss(batch_y, preds)
        tl += l
        lm.backward(g)
        optimizer.step()
        optimizer.zero_grad()
        if step != steps - 1:
            bar.update(step + 1, values=[("train_loss", l)])
        else:
            val_ppx = perplexity(idx_for_ppx, lm)
            tl /= steps
            metrics = {"train_loss": tl, "val_ppx": val_ppx}
            params = None
            model_file = None
            if best_val is None or best_val > val_ppx:
                patience_count = 0
                best_val = val_ppx
                model_dir = BASE_DIR + "models_{}/".format(ACTIVATION)
                make_directory(model_dir)
                model_file = model_dir + "model_hidden_%d_epoch_%d_ppx_%.2f.model" % (NUM_HIDDEN, epoch + 1, val_ppx)
                params = lm
            else:
                patience_count += 1
                if patience_count > PATIENCE:
                    patience_count = 0
                    optimizer.lr /= 2
            history = History(epoch + 1, metrics, params, model_file)
            s.add_history(history)
            bar.update(step + 1, values=[("train_loss", l), ("val_ppx", val_ppx), ("lr", optimizer.lr)])
# ======== End of Training Loop ===========#
s.save(BASE_DIR + "Summary/summary_activation_{}_Hidden_{}_valppx_{}.pkl".format(ACTIVATION, NUM_HIDDEN, best_val))
