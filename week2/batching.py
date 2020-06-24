"""
This module contains the class that handles minibatching data. Will automatically shuffle the dataset at each
epoch.
"""
import numpy as np
from copy import  deepcopy

class MiniBatcher:
    """A class meant to handle to batching of a given dataset

    Randomly samples the dataset via an index-yielder
    Simply shuffling the data adds complications because we need our labels to always correspond to our inputs.
    Instead, the class holds an attribute that is shuffled independently of the data itself. This attribute is then
    sliced to size batch_size which is then used to index the given data and labels.

    At the end of each epoch, when the MiniBatcher is out of data, it will return (False, False)
    To reset the batching, simply call MiniBatcher.new_epoch(). This method will auto shuffle the indexes once again
    and you will be able to pull out another batch.

    Attributes:
        User inputted:
            data (array): our data samples
            labels (array): our labels
            batch_size (int): the size of our minibatches
            seed (int): the random seed to allow for experiment reproducibility

        Class managed:
            random_generator (RandomState): random number generator that inherits the user-inputted seed
            data_batching_order (array): an array of numbers 0-len(data)
            __mb_idx__ (generator): a generator that will yield an array of indexes
    """

    def __init__(self, data, labels, batch_size, seed):
        self.data = deepcopy(data)
        self.labels = deepcopy(labels)
        self.batch_size = batch_size
        self.seed = seed

        self.random_generator = np.random.RandomState(seed)

        # This minibatch class works off of a generator that yields an array of indexes
        # This works for us by slicing the data_batching_order array
        # (1) Init array with every idx from [0,len(data)]
        # (2) Shuffle said array
        # (3) Init our generator
        self.data_batching_order = np.array([x for x in range(data.shape[0])])
        self._data_shuffle()
        self.__mb_idx__ = self._idx_yielder()

    def fetch_minibatch(self):
        """
        If this method returns [False, False], then we know that our data is out and it's time to start a new epoch!
        """
        # As our generator can run out we need a try/except
        try:
            idxs = self.__mb_idx__.__next__()
            return self.data[idxs], self.labels[idxs]
        except:
            return False, False

    def _data_shuffle(self):
        """Simply shuffles the data_batching_order attribute in-place"""
        # This performs the shuffle in place, no need to return anything
        self.random_generator.shuffle(self.data_batching_order)

    def _idx_yielder(self):
        """Returns a generator of a sliding window over self.data_batching_order
        """
        lag_idx = 0
        lead_idx = 0 + self.batch_size

        while lag_idx <= self.data.shape[0]:
            yield self.data_batching_order[lag_idx: lead_idx]
            lag_idx += self.batch_size
            lead_idx += self.batch_size

    def new_epoch(self):
        """At the beginning of every new epoch, call this method to reset the generator. Will be different than the last."""
        self._data_shuffle()
        self.__mb_idx__ = self._idx_yielder()
