import cPickle as cp
import torch
import os
import pdb


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
