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
        d = data[ix].split(' ')
        d = [START_TOK] + d + [END_TOK]
        for jx in xrange(len(d) - N_GRAM):
            ngram = [vocab[w] if w in vocab else vocab["<UNK>"] for w in d[jx: jx + N_GRAM]]
            pred = vocab[d[jx + N_GRAM]] if d[jx + N_GRAM] in vocab else vocab["<UNK>"]
            ret_data.append((ngram, pred))
    return ret_data


# ========== Computing Perplexity ========#
def perplexity(index, model):
    """
    Compute the perplexity of the validation data
        :param index: Indexed data
        :param model: The model to evaluate
        :return ppx: The perplexity
    """
    batch = 512
    ppx = 0.
    M = len(index)
    for ix in xrange(0, len(index), batch):
        batch_x, batch_y = map(lambda x: np.array(x), zip(*index[ix: ix + batch]))
        preds = model(batch_x)
        lp = np.log2(preds[range(preds.shape[0]), batch_y]).sum()
        ppx += lp
    return 2 ** (-ppx / M)


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
ix_train = index_data(training_data, vocab)
ix_val = index_data(val_data, vocab)
ix_val_x, ix_val_y = map(lambda x: np.array(x), zip(*ix_val))
loss = categorical_cross_entropy()
optimizer = SGD(lm.parameters(), lr=LR, momentum=0.9)
patience_count = 0
for epoch in xrange(N_EPOCHS):
    print "\nEPOCH ({}/{})".format(epoch + 1, N_EPOCHS)
    steps = -(-len(ix_train) // BATCH_SIZE)  # Round up
    bar = Progbar(steps)
    tl = 0.
    tppx = 0.
    for step, ix in enumerate(xrange(0, len(ix_train), BATCH_SIZE)):
        batch_x, batch_y = map(lambda x: np.array(x), zip(*ix_train[ix: ix + BATCH_SIZE]))
        batch_x = np.array(batch_x)
        preds = lm(batch_x)
        l, g = loss(batch_y, preds)
        lm.backward(g)
        optimizer.step()
        optimizer.zero_grad()
        tl += l
        ppx = perplexity(ix_train[ix: ix + BATCH_SIZE], lm)
        tppx += ppx
        if step != steps - 1:
            bar.update(step + 1, values=[("train_loss", l), ("train_ppx", ppx)])
        else:
            val_ppx = perplexity(ix_val, lm)
            val_loss = 0.
            for jx in xrange(0, len(ix_val_x), BATCH_SIZE):
                val_loss += loss(ix_val_y[jx: jx + BATCH_SIZE], lm(ix_val_x[jx: jx + BATCH_SIZE]), normalize=False)[0]
            val_loss /= len(ix_val_x)
            tl /= steps
            tppx /= steps
            metrics = {"train_loss": tl, "train_ppx": tppx, "val_loss": val_loss, "val_ppx": val_ppx}
            params = None
            model_file = None
            if best_val is None or best_val > val_ppx:
                patience_count = 0
                best_val = val_ppx
                model_dir = BASE_DIR + "models_{}/".format(ACTIVATION)
                make_directory(model_dir)
                model_file = model_dir + "model_embed_%d_hidden_%d_epoch_%d_ppx_%.2f.model" % (NUM_EMBED, NUM_HIDDEN, epoch + 1, val_ppx)
                params = lm
            else:
                patience_count += 1
                if patience_count > PATIENCE:
                    patience_count = 0
                    optimizer.lr /= 2
            history = History(epoch + 1, metrics, params, model_file)
            s.add_history(history)
            bar.update(step + 1, values=metrics.items() + [('lr', optimizer.lr)])
# ======== End of Training Loop ===========#
summary_dir = BASE_DIR + "Summary_NGRAM/"
make_directory(summary_dir)
s.save(summary_dir + "summary_activation_{}_Hidden_{}_valppx_{}.pkl".format(ACTIVATION, NUM_HIDDEN, best_val))
