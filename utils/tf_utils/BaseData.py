import tensorflow as tf
import pandas as pd
import numpy as np

class BaseData(object):
    def __init__(self,flags):
        self.flags = flags
        self.QUEUE_CAPACITY = 500
        self.SHUFFLE_MIN_AFTER_DEQUEUE = self.QUEUE_CAPACITY // 5

    def write_tfrecord(self):
        raise NotImplementedError()

    def _read_and_decode_single_example(self):
        raise NotImplementedError()

    def _batching(self,x,y):
        # methods are determined by flags.task
        """
            Input:
                x,y: [F1,F2,..] single tensors
            Return:
                xs,ys: [B,F1,F2..] batched tensors
        """
        raise NotImplementedError()
