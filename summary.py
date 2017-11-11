import cPickle as cp
import os


class Summary(object):
    def __init__(self, meta):
        """
        The information regarding a run
        """
        for elem in meta:
            setattr(self, elem, meta[elem])
        self.histories = []

    def add_history(self, history):
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
        best = True
        if weights_file is not None:
            directory = '/'.join(weights_file.split('/')[:-1])
            for file in os.listdir(directory):
                ppx = float(file.split("_")[-1].strip(".model"))
                if ppx > metrics['val_ppx']:
                    os.remove(directory + "/" + file)
                else:
                    best = False
                    break
            if best:
                self.params = weights_file
                cp.dump(params, open(weights_file, "wb"))
