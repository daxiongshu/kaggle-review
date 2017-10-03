import tensorflow as tf
from comps.personal.personal_db import personalDB
from comps.personal.baobao.d2v import D2V

def train_nn(flags):
    myDB = personalDB(flags)
    model = BaoNN(flags,myDB)
    model.train()


def predict_nn(flags):
    myDB = personalDB(flags)
    model = BaoNN(flags,myDB)
    model.predict()

class BaoNN(D2V):

    def train(self):
        self.train_from_placeholder(va=True) # this will call _build()

    def predict(self):
        self.predict_from_placeholder(activation="softmax")

    def set_train_var(self):
        # This function could be overwritten
        self.var_list = [var for var in tf.trainable_variables() if "D2V" not in var.name]

    def _build(self):
        V = self.V
        M = self.flags.embedding_size # 64
        H = self.flags.num_units 
        C = self.flags.classes
        D = self.flags.d2v_size # embedding for d2v

        netname = "D2V"
        with tf.variable_scope(netname):
            self.inputs = tf.placeholder(dtype=tf.int32,shape=[None]) #[B]
            layer_name = "{}/embedding".format(netname)
            x = self._get_embedding(layer_name, self.inputs, V, D, reuse=False) # [B, S, M]

        netname = "NN"
        cell_name = self.flags.cell
        H1,H2 = 32,16
        with tf.variable_scope(netname):
            net = self._fc(x, fan_in=D, fan_out=H1, layer_name="%s/fc1"%netname, activation='relu')
            net = self._dropout(net)
            net = self._fc(net, fan_in=H1, fan_out=H2, layer_name="%s/fc2"%netname, activation='relu')
            net = self._dropout(net)
            net = self._fc(net, fan_in=H2, fan_out=C, layer_name="%s/fc3"%netname, activation=None)
            self.logit = net
        
    def _batch_gen(self):
        from random import sample,randint
        self.DB.get_split()
        epochs = self.flags.epochs
        fold = self.flags.fold

        if fold>=0:
            docs_ids = list(self.DB.split[fold][0])
        else:
            docs_ids = list(range(self.DB.data['training_text'].shape[0]))

        B = min(self.flags.batch_size,len(docs_ids))
        batches_per_epoch = len(docs_ids)//B

        y = self.DB.y
        #print(batches_per_epoch)
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                for idx in sample(docs_ids,B):
                    inputs.append(idx+1)
                    labels.append(y[idx])
                yield inputs, labels, epoch 

    def _batch_gen_test(self):
        return self._batch_gen_va()

    def _batch_gen_va(self):
        from random import sample,randint
        self.DB.get_split()
        fold = self.flags.fold

        if fold>=0:
            docs_ids = list(self.DB.split[fold][1])
        else:
            bias = self.DB.data['training_text'].shape[0] 
            docs_ids = list(range(bias,bias+self.DB.data['test_text_filter'].shape[0]))

        B = min(self.flags.batch_size,len(docs_ids))
        batches_per_epoch = len(docs_ids)//B

        y = self.DB.y
        #print(batches_per_epoch)
        N = len(docs_ids)
        B = min(self.flags.batch_size,N//2)
        for s in range(0,len(docs_ids),B):
            inputs = []
            labels = [] # 0 or 1
            e = min(s+B,N)
            for idx in docs_ids[s:e]:
                inputs.append(idx+1)
                if fold>=0:
                    labels.append(y[idx])
            yield inputs, labels, 1
