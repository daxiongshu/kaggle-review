import tensorflow as tf
import numpy as np
import pandas as pd
import time
from models.tf_models.BaseModel import BaseModel

class BaseRnnModel(BaseModel):

    def __init__(self,flags):
        super().__init__(flags)
        tf.GraphKeys.EMBEDDINGS = 'embeddings'

    def _get_initial_lstm(self, layer_name, features):
        with tf.variable_scope(layer_name.split('/')[-1]):
            w1 = self._get_variable(layer_name, name='w_h', shape=[self.D, self.H])
            b1 = self._get_variable(layer_name, name='b_h', shape=[self.H])
            w2 = self._get_variable(layer_name, name='w_c', shape=[self.D, self.H])
            b2 = self._get_variable(layer_name, name='b_c', shape=[self.H])

            if features is not None:
                features_mean = tf.reduce_mean(features, 1)
            else:
                features_mean = self._get_variable(layer_name, name='random_mean', shape=[self.flags.batch_size, self.D]) 
            h = tf.nn.tanh(tf.matmul(features_mean, w1) + b1)
            c = tf.nn.tanh(tf.matmul(features_mean, w2) + b2)
        return c, h


    def _get_embedding(self, layer_name, inputs, v,m,reuse=False):
        """
            V: vocabulary size
            M: embedding sze
        """
        with tf.variable_scope(layer_name.split('/')[-1], reuse=reuse):
            w = self._get_variable(layer_name, name='w', shape=[v, m])
            c = tf.zeros([1,m])
            w = tf.concat([c,w],axis=0)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x


    def _get_rnn_cell(self, cell_name, num_units, activation=tf.tanh,use_peepholes=False):
        if cell_name == "BASIC_LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units,activation=activation)
        elif cell_name == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=num_units,activation=activation)
        elif cell_name == "LSTM":
            cell = tf.contrib.rnn.LSTMCell(num_units=num_units,activation=activation,use_peepholes=use_peepholes)
        elif cell_name == "BLOCK_LSTM":
            cell = tf.contrib.rnn.LSTMBlockCell(num_units=num_units)
        elif cell_name == "BLOCK_GRU":
            cell = tf.contrib.rnn.GRUBlockCell(num_units)
        elif cell_name == "NAS":
            cell = tf.contrib.rnn.NASCell(num_units)
        else:
            print("Unknown cell name", cell_name)
            assert 0

        return cell

    def _dynamic_padding(self, seqs, val=0):
        # B,S,...
        maxl = 0
        ll = isinstance(seqs[0][0],list)
        for seq in seqs:
            maxl = max(len(seq),maxl)
        for c,seq in enumerate(seqs):
            if ll:
                lu = len(seqs[0][0])
                seqs[c].extend([[val]*lu]*(maxl - len(seq)))
            else:
                seqs[c].extend([val]*(maxl - len(seq)))
        return seqs

