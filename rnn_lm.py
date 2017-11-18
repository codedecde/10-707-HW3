'''
The Recurrent Language Model
'''


import cPickle as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from constants import *
from summary import Summary, History
import pdb
from utils import Progbar, make_directory, get_np_tensor
from RNNLanguageModel import RNNLanguageModel
import argparse
import numpy as np
from torch.autograd import Variable


use_cuda = torch.cuda.is_available()
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
def perplexity(index, model, batch):
    """
    Compute the perplexity of the validation data
        :param index: Indexed data
        :param model: The model to evaluate
        :return ppx: The perplexity
    """
    def fn(x):
        return Variable(torch.LongTensor(np.array(x))).cuda() if use_cuda else Variable(torch.LongTensor(np.array(x)))
    ppx = 0.
    M = len(index)
    for ix in xrange(0, len(index), batch):
        batch_x, batch_y = map(fn, zip(*index[ix: ix + batch]))
        preds = model(batch_x)
        lp = -F.nll_loss(preds, batch_y, size_average=False) / np.log(2)
        ppx += lp
    ppx = ppx.cpu().data.numpy()[0] if use_cuda else ppx.data.numpy()[0]
    return 2 ** (-ppx / M)


# ========== Argument Parser ============#
def parse_args():
    def check_boolean(args, bool_lst):
        for a in bool_lst:
            assert hasattr(args, a) and getattr(args, a).lower() in set(["true", "false"]), "Error, boolean encountered with value %s" % getattr(args, a)
            setattr(args, a, True if getattr(args, a).lower() == 'true' else False)
        return args
    parser = argparse.ArgumentParser(description="Language Model")
    parser.add_argument("-ne", "--embed_size", help="Embedding Size", dest="n_embed", default=16, type=int)
    parser.add_argument("-nd", "--hidden_size", help="Hidden Size", dest="n_dim", default=128, type=int)
    parser.add_argument("-epochs", "--num_epochs", help="Number of epochs", dest="n_epochs", default=100, type=int)
    parser.add_argument("-b", "--batch_size", help="Batch Size", dest="batch", default=16, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning Rate", dest="lr", default=0.1, type=float)
    parser.add_argument("-p", "--patience", help="Patience", dest="patience", default=2, type=int)
    parser.add_argument("-t", "--truncate", help="Truncate BPTT", dest="truncate", type=int)
    parser.add_argument("-o", "--ortho", help="Orthogonal initialization", dest="ortho", default="False", type=str)
    args = parser.parse_args()
    args = check_boolean(args, ["ortho"])
    return args


args = parse_args()
NUM_EMBED = args.n_embed
NUM_HIDDEN = args.n_dim
N_EPOCHS = args.n_epochs
BATCH_SIZE = args.batch
LR = args.lr
PATIENCE = args.patience
TRUNCATE = args.truncate
ORTHO = args.ortho
# ========== Initialize Summary ==========#
meta = {"NUM_EMBED": NUM_EMBED,
        "NUM_HIDDEN": NUM_HIDDEN,
        "N_EPOCHS": N_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "TRUNCATE": TRUNCATE,
        "ORTHO": ORTHO,
        "LR": LR}
for elem in meta:
    print "{}:\t{}".format(elem, meta[elem])
s = Summary(meta)
best_val = None
# ========== Training Loop ===============#
lm = RNNLanguageModel(len(vocab), NUM_EMBED, NUM_HIDDEN, ortho=ORTHO)
if use_cuda:
    lm = lm.cuda()
training_data = map(lambda x: x.lower().strip(), open(TRAIN_FILE).readlines())
val_data = map(lambda x: x.lower().strip(), open(VAL_FILE).readlines())
ix_train = index_data(training_data, vocab)
ix_val = index_data(val_data, vocab)
ix_val_x, ix_val_y = map(lambda x: Variable(torch.LongTensor(np.array(x))), zip(*ix_val))
ix_val_x, ix_val_y = (ix_val_x.cuda(), ix_val_y.cuda()) if use_cuda else (ix_val_x, ix_val_y)

loss = nn.NLLLoss()
optimizer = optim.SGD(lm.parameters(), lr=LR, momentum=0.9)
patience_count = 0
lr = LR
trunk = 0 if TRUNCATE is None else TRUNCATE
for epoch in xrange(N_EPOCHS):
    print "\nEPOCH ({}/{})".format(epoch + 1, N_EPOCHS)
    steps = -(-len(ix_train) // BATCH_SIZE)  # Round up
    bar = Progbar(steps)
    tl = 0.
    tppx = 0.
    for step, ix in enumerate(xrange(0, len(ix_train), BATCH_SIZE)):
        batch_x, batch_y = map(lambda x: np.array(x), zip(*ix_train[ix: ix + BATCH_SIZE]))
        batch_x = Variable(torch.LongTensor(np.array(batch_x)))
        batch_y = Variable(torch.LongTensor(np.array(batch_y)))
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        if np.random.rand() < 0.1:
            # Truncate BPTT for 10% cases when specified
            preds = lm(batch_x, TRUNCATE)
        else:
            preds = lm(batch_x, None)
        l = loss(preds, batch_y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        tl += l
        ppx = perplexity(ix_train[ix: ix + BATCH_SIZE], lm, BATCH_SIZE)
        tppx += ppx
        if step != steps - 1:
            l_data = get_np_tensor(l)[0]
            bar.update(step + 1, values=[("train_loss", l_data)])
        else:
            val_ppx = perplexity(ix_val, lm, BATCH_SIZE)
            val_loss = 0.
            for jx in xrange(0, len(ix_val_x), BATCH_SIZE):
                val_loss += F.nll_loss(lm(ix_val_x[jx: jx + BATCH_SIZE]), ix_val_y[jx: jx + BATCH_SIZE], size_average=False)
            tl /= steps
            tl = get_np_tensor(tl)[0]
            tppx /= steps
            val_loss /= ix_val_x.size(0)
            val_loss = get_np_tensor(val_loss)[0]
            metrics = {"train_loss": tl, "train_ppx": tppx, "val_loss": val_loss, "val_ppx": val_ppx}
            params = None
            model_file = None
            if best_val is None or best_val > val_ppx:
                patience_count = 0
                best_val = val_ppx
                model_dir = BASE_DIR + "models_RNN/"
                make_directory(model_dir)
                model_file = model_dir + "model_ortho_%s_truncated_%d_embed_%d_hidden_%d_epoch_%d_ppx_%.2f.model" % (str(ORTHO), trunk, NUM_EMBED, NUM_HIDDEN, epoch + 1, val_ppx)
                params = lm
            else:
                patience_count += 1
                if patience_count > PATIENCE:
                    patience_count = 0
                    lr /= 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            history = History(epoch + 1, metrics, params, model_file)
            s.add_history(history, pytorch=True)
            bar.update(step + 1, values=[("train_loss", tl), ("val_ppx", val_ppx), ("lr", lr)])
# ======== End of Training Loop ===========#
summary_dir = BASE_DIR + "Summary_RNN/"
make_directory(summary_dir)
s.save(summary_dir + "summary_ortho_%s_truncated_{}_Hidden_{}_Embed_{}_valppx_{}.pkl".format(str(ORTHO), trunk, NUM_HIDDEN, NUM_EMBED, best_val))
