import tensorflow as tf
from comps.personal.baobao.embedding import BaoEmbedding
from comps.personal.personal_db import personalDB
import os
import numpy as np

def train_d2v(flags):
    myDB = personalDB(flags)
    model = D2V(flags,myDB)
    model.train()

def show_d2v(flags):
    myDB = personalDB(flags)
    model = D2V(flags,myDB)
    model.show_embedding("D2V/embedding/w:0")

class D2V(BaoEmbedding):

    def set_train_var(self):
        # This function could be overwritten
        self.var_list = [var for var in tf.trainable_variables() if "CBOW" not in var.name]

    def _build(self):
        W = self.flags.window_size
        M = self.flags.embedding_size
        D = self.flags.d2v_size # embedding for d2v
        V = self.V # vocabulary size, should be passed from DB
        H = 128

        # the real window is W*2 + 1
        self.inputs = tf.placeholder(tf.int32, shape=(None,2)) # [B, 2]
        word,doc = tf.split(self.inputs,[1,1],axis=1)
        netname = "CBOW"
        with tf.variable_scope(netname):
            layer_name = "{}/embedding".format(netname)
            x1 = self._get_embedding(layer_name, word, V, M, reuse=False) # [B, 1, M]

        netname = "D2V"
        with tf.variable_scope(netname):
            layer_name = "{}/embedding".format(netname)
            x2 = self._get_embedding(layer_name, doc, V, D, reuse=False) # [B, 1, D]
            x = tf.concat([x1,x2],axis=2)
            x = tf.squeeze(x) #[B,D+M]

            layer_name = "{}/fc1".format(netname)
            net = self._fc(x, fan_in=D+M, fan_out=H, layer_name=layer_name,
                    activation='relu') # [B, H]

            layer_name = "{}/fc2".format(netname)
            net = self._fc(net, fan_in=H, fan_out=2, layer_name=layer_name,
                    activation=None) # [B, 2]

            self.logit = net

    def _batch_gen(self):
        from random import sample,randint

        all_words = list(self.all_words)
        clean_doc = self.DB.clean_doc

        epochs = self.flags.epochs
        B = self.flags.batch_size
        w2id = self.DB.w2id

        W = self.flags.window_size
        V = self.V # vocabulary size, should be passed from DB

        docs = clean_doc['training_text'] + clean_doc['test_text_filter']
        doc_dics = [set(doc) for doc in docs]
        docs_ids = range(len(docs))
        docs_len = [len(doc) for doc in docs]
        B = min(B,len(docs))
        batches_per_epoch = sum([l//B for l in docs_len])

        #from utils.draw.sns_draw import distribution
        #distribution(docs_len)

        print(batches_per_epoch)
        neg_dic = {}
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                for idx in sample(docs_ids,B):
                    #assert idx>0
                    doc = docs[idx]
                    doc_len = len(doc)
                    target_id = randint(0,doc_len-1)
                    pos = [w2id[doc[target_id]],idx+1]
                   
                    if idx not in neg_dic:
                        neg_words = [i for i in w2id if i not in doc_dics[pos[1]-1]]
                        neg_dic[idx] = neg_words
                    else:
                        neg_words = neg_dic[idx]
                    assert len(neg_words)>0
                    target_id = randint(0,len(neg_words)-1)
                    neg = [w2id[neg_words[target_id]],idx+1]

                    inputs.extend([pos,neg])
                    labels.extend([[0,1],[1,0]])
                yield inputs, labels, epoch


    def _write_meta(self):
        path = os.path.join(self.flags.log_path, 'metadata.tsv')
        if os.path.exists(path):
            return
        clean_doc = self.DB.clean_doc
        docs = clean_doc['training_text'] + clean_doc['test_text_filter']
        docs_ids = range(len(docs))
        with open(path,'w') as fo:
            for i in docs_ids:
                if i<len(self.DB.y):
                    fo.write('{}-{}\n'.format(i,1+np.argmax(self.DB.y[i])))
                else:
                    fo.write('{}\n'.format(i))
            fo.write("__\n")
