import tensorflow as tf
from models.tf_models.BaseRnnModel import BaseRnnModel
from comps.personal.personal_db import personalDB
import os

def train_embedding(flags):
    myDB = personalDB(flags,name='full')
    model = BaoEmbedding(flags,myDB)
    model.train() 

def show_embedding(flags):
    myDB = personalDB(flags,name='full')
    model = BaoEmbedding(flags,myDB)
    model.show_embedding("CBOW/embedding/w:0")

class BaoEmbedding(BaseRnnModel):

    def __init__(self,flags,DB):
        super().__init__(flags)
        self.DB = DB
        all_words = self.DB.select_top_words(['training_text','test_text_filter'],"Text",mode='tfidf')
        self.DB.get_words(all_words)
        self.DB.get_clean_doc(['training_text','test_text_filter'],"Text",all_words)
        self.all_words = all_words
        self.V = len(all_words)


    def _write_meta(self):
        path = os.path.join(self.flags.log_path, 'metadata.tsv')
        if os.path.exists(path):
            return
        with open(path,'w') as fo:
            for i in range(1,len(self.DB.id2w)):
                fo.write('{}\n'.format(self.DB.id2w[i]))
            fo.write("__\n")

    def train(self):
        self.train_from_placeholder() # this will call _build()

    def _build(self):
        netname = "CBOW"
        W = self.flags.window_size
        M = self.flags.embedding_size
        V = self.V # vocabulary size, should be passed from DB
        H = 128

        # the real window is W*2 + 1
        with tf.variable_scope(netname):
            self.inputs = tf.placeholder(tf.int32, shape=(None,W*2+1)) # [B, W*2+1]

            layer_name = "{}/embedding".format(netname)
            x = self._get_embedding(layer_name, self.inputs, V, M, reuse=False) # [B, W*2+1, M]
            x = tf.reshape(x,[tf.shape(x)[0],tf.shape(x)[1]*tf.shape(x)[2]]) # [B,(W*2+1)*M]

            layer_name = "{}/fc1".format(netname)
            net = self._fc(x, fan_in=M*(W*2+1), fan_out=H, layer_name=layer_name, 
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
        w2id = self.DB.w2id

        W = self.flags.window_size
        V = self.V # vocabulary size, should be passed from DB

        docs = clean_doc['training_text'] + clean_doc['test_text_filter']
        docs_ids = range(len(docs))
        docs_len = [len(doc) for doc in docs]
        B = min(self.flags.batch_size,len(docs))
        batches_per_epoch = sum([l//B for l in docs_len])

        from utils.draw.sns_draw import distribution
        distribution(docs_len)

        print(batches_per_epoch)
        for epoch in range(epochs):           
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                for idx in sample(docs_ids,B):
                    doc = docs[idx]
                    doc_len = len(doc)
                    target_id = randint(0,doc_len-1)
                    pos = doc[target_id-W : target_id+W+1]
                    pos = [self.DB.w2id[word] for word in pos]
                    if target_id<W:
                        pos = [0]*(W*2+1-len(pos)) + pos
                    elif target_id + W >= doc_len:
                        pos = pos + [0]*(W*2+1-len(pos))
                    neg = [i for i in pos]
                    neg[W] = self.DB.w2id[all_words[randint(0,V-1)]]
                    assert len(pos) == W*2+1
                    assert len(neg) == W*2+1
                    inputs.extend([pos,neg])
                    labels.extend([[0,1],[1,0]])
                yield inputs, labels, epoch




