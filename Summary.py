import cPickle as cp
import torch
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from constants import *
import matplotlib
import os
import pdb
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Summary(object):
    def __init__(self, meta, monitor="val_ppx"):
        """
        The information regarding a run
        """
        for elem in meta:
            setattr(self, elem, meta[elem])
        self.histories = []
        self.best_params = None
        self.monitor = monitor

    def add_history(self, history, pytorch=False):
        if self.best_params is None or self.best_params[0] > history.metrics[self.monitor]:
            assert history.weights_file is not None, "Weights file for best history is None"
            assert history.params is not None, "Params for best history is None"
            if self.best_params is not None:
                os.remove(self.best_params[1])
            if not pytorch:
                cp.dump(history.params, open(history.weights_file, "wb"))
            else:
                state_dict = history.params.state_dict()
                if torch.cuda.is_available():
                    state_dict = {w: state_dict[w].cpu() for w in state_dict}
                torch.save(state_dict, history.weights_file)
            self.best_params = (history.metrics[self.monitor], history.weights_file)
        del history.params
        self.histories.append(history)

    def save(self, filename):
        meta = {}
        for elem in self.__dict__:
            if elem != 'histories':
                meta[elem] = getattr(self, elem)
        data = {'meta': meta, 'history': self.histories}
        cp.dump(data, open(filename, 'wb'))


class History(object):
    def __init__(self, epoch, metrics, params, weights_file):
        self.epoch = epoch
        self.metrics = metrics
        self.weights_file = weights_file
        self.params = params


def plot_summaries(summaries, headers, save_file, ylabel):
    """
    Compares the summaries to each other based on headers
        :param summaries: [(name, summary)]
        :param headers: [string]: The headers to compare
        :param save_file: string: Where to save the image
        :param ylabel: string: The y axis label
    """
    colors = ["#ffc300", "#d00000", "#bada55", "#1c79e1", "#f15037", "#0e2f44", "#000080", "#9f96ff"]
    names, summaries = map(list, zip(*summaries))
    plot_dict = OrderedDict()
    for name in names:
        for header in headers:
            key = "{}: {}".format(name, header)
            plot_dict[key] = []
    for ix in xrange(len(summaries)):
        name = names[ix]
        for hist in summaries[ix]['history']:
            for header in headers:
                key = "{}: {}".format(name, header)
                plot_dict[key].append(hist.metrics[header])
    epochs = range(1, summaries[0]['history'][-1].epoch + 1)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt_objects = OrderedDict()
    for jx, key in enumerate(plot_dict):
        if jx < len(colors):
            plt_objects[key], = plt.plot(epochs, np.array(plot_dict[key]), label=key, color=colors[jx])
        else:
            plt_objects[key], = plt.plot(epochs, np.array(plot_dict[key]), label=key)
    plt.legend(plt_objects.values(), plt_objects.keys(), loc=0)
    plt.savefig(save_file)
    plt.close()
    return plot_dict


def generate_csv(summaries, filename):
    """
    Generates CSV given a list of summaries
        :param summaries: [summary]: list of summary instances
        :param filename: String: The filename
    """
    with open(filename, 'wb') as f:
        header = ','.join(['ACTIVATION', 'HIDDEN SIZE', 'TRAIN LOSS', 'VAL LOSS', 'TRAIN PPX', 'VAL PPX']) + '\n'
        f.write(header)

        def extract_best(summary, metric):
            return min([h.metrics[metric] for h in summary['history']])
        for summary in summaries:
            activation = summary['meta']['ACTIVATION']
            h_size = summary['meta']['NUM_HIDDEN']
            train_loss, val_loss, train_ppx, val_ppx = extract_best(summary, 'train_loss'), extract_best(summary, 'val_loss'), extract_best(summary, 'train_ppx'), extract_best(summary, 'val_ppx')
            line = ",".join([activation] + map(lambda x: "%.2f" % (x), [h_size, train_loss, val_loss, train_ppx, val_ppx])) + '\n'
            f.write(line)


def generate_language(sent, vocab, model, end_tok=END_TOK):
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
        if ix2vocab[pred] == end_tok:
            break
    return ' '.join([ix2vocab[jx] for jx in gen_s])


def distance(v1, v2):
    return ((v1 - v2) ** 2).sum()


def find_nearest_k(word, vocab, model, k=5):
    """
    Finds the k nearest neighbors based on euclidien distance
    """
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


def visualize_embeddings(matrix, vocab, file, num_words=50):
    """
    Generate the scatterplot for the matrix
    """
    ix2vocab = {vocab[w]: w for w in vocab}
    indices = np.random.choice(range(matrix.shape[0]), replace=False, size=num_words)
    small_matrix = matrix[indices]
    labels = [ix2vocab[index] for index in indices]
    plt.scatter(small_matrix[:, 0], small_matrix[:, 1])
    for i, txt in enumerate(labels):
        plt.annotate(txt, (small_matrix[i, 0], small_matrix[i, 1]), xytext=(3, 3),
                     textcoords="offset points", ha='left', va='top')
    plt.savefig(file)
    plt.close()
