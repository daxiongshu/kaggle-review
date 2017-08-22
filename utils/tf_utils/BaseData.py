import tensorflow as tf
import pandas as pd
import numpy as np

class BaseData(object):
    def __init__(self,flags):
        self.flags = flags

    def write_tfrecord(self):
        raise NotImplementedError()

    def _read_and_decode_single_example(self):
        raise NotImplementedError()

    def batch_generator(self, is_onehot):
        if 'train' in self.flags.task:
            return self._batch_generator_train(is_onehot)
        else:
            return self._batch_generator_predict(is_onehot)

    def _batch_generator_train(self, is_onehot):
        raise NotImplementedError()

    def _batch_generator_predict(self, is_onehot): 
        raise NotImplementedError()

