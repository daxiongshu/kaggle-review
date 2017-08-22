"""
Base class for sequence data
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from utils.tf_utils.BaseData import BaseData
from utils.tf_utils.utils import _int64_feature, _bytes_feature

class BaseSeqData(BaseData):
    def __init__(self,flags):
        super().__init__(flags)        
        self.QUEUE_CAPACITY = 500
        self.SHUFFLE_MIN_AFTER_DEQUEUE = self.QUEUE_CAPACITY // 5

    def _count_records(self, file_list):
        """Returns number of records in files from file_list."""
        num_records = 0
        for tfrecord_file in file_list:
            tf.logging.info('Counting records in %s.', tfrecord_file)
            for _ in tf.python_io.tf_record_iterator(tfrecord_file):
                num_records += 1
        tf.logging.info('Total records: %d', num_records)
        return num_records

    def _shuffle_inputs(self, input_tensors, capacity, 
        min_after_dequeue, num_threads):

        """Shuffles tensors in `input_tensors`, maintaining grouping."""
        shuffle_queue = tf.RandomShuffleQueue(
            capacity, min_after_dequeue, dtypes=[t.dtype for t in input_tensors])
        enqueue_op = shuffle_queue.enqueue(input_tensors)
        runner = tf.train.QueueRunner(shuffle_queue, [enqueue_op] * num_threads)
        tf.train.add_queue_runner(runner)

        output_tensors = shuffle_queue.dequeue()

        for i in range(len(input_tensors)):
            output_tensors[i].set_shape(input_tensors[i].shape)

        return output_tensors

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

