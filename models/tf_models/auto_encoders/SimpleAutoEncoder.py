import tensorflow as tf
from models.tf_models.auto_encoders.BaseAutoEncoder import BaseAutoEncoder

class SimpleAutoEncoder(BaseAutoEncoder):
    """
    only fc layers
    """
    def _config(self):
        #self.input_dim = None
        #self.encode_layers = None
        #self.decode_layers = None
        raise NotImplementedError()

    def _build(self):
        self._config()
        if self.input_dim != self.decode_layers[-1]:
            print("decode output dim != input dim",self.decode_layers[-1],self.input_dim)
            assert 0
        net_name = "simple_auto_encoder"
        with tf.variable_scope(net_name):
            self._encoder("%s/encoder"%net_name)
            self._decoder("%s/decoder"%net_name)
        if "predict" in self.flags.task:
            self.logit = self.code

    def train(self):
        self.train_from_placeholder() # this will call _build()        

    def predict(self):
        self.predict_from_placeholder() # this will call _build()
    def _encoder(self,name):
        self.inputs = tf.placeholder(tf.float32, shape=(None,self.input_dim))
        self.code = self._fc_block(self.inputs, name, ['relu']*len(self.encode_layers), self.encode_layers)

    def _decoder(self,name):
        self.logit = self._fc_block(self.code, name, ['relu']*len(self.decode_layers), self.decode_layers)
