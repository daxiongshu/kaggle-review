import tensorflow as tf
from models.tf_models.BaseModel import BaseModel

class BaseAutoEncoder(BaseModel):

    def _encoder(self):
        # this func builds self.code
        raise NotImplementedError()

    def _dencoder(self):
        # this func builds self.logit
        raise NotImplementedError()

    def _get_loss(self,labels):

        with tf.name_scope("Loss"):
            with tf.name_scope("reconstruction_loss"):
                labels = tf.cast(labels, tf.float32)
                self.loss = tf.reduce_mean(tf.pow(labels - self.logit, 2))

    def _fc_block(self, net, name, activations, layer_sizes, batchnorm=None):
        assert len(activations) == len(layer_sizes)
        if batchnorm is None:
            batchnorm = [1 for i in activations]

        with tf.variable_scope(name.split('/')[-1]):
            for i in range(len(layer_sizes)):
                net = self._fc(net, fan_in=net, fan_out=layer_sizes[i], layer_name, activation=activations[i], L2=1, use_bias=True)

                if batchnorm[i]:
                    net = self._batch_normalization(net, layer_name='%s/batch_norm%d'%(name,i))


        return net

