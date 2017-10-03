import tensorflow as tf
from comps.personal.personal_db import personalDB
from comps.personal.baobao.gene_text_db import geneTextDB
from comps.personal.baobao.embedding import BaoEmbedding
from models.tf_models.BaseCnnModel import BaseCnnModel
import os
import pandas as pd
import numpy as np

def train_cnn(flags):
    myDB = personalDB(flags,name='full')
    gDB = geneTextDB(flags,W=10,bar=0)
    model = BaoCnn(flags,myDB,gDB)
    model.train()

def test_cnn(flags):
    myDB = personalDB(flags,name='full')
    gDB = geneTextDB(flags,W=10,bar=0)
    model = BaoCnn(flags,myDB,gDB)
    model.predict()

class BaoCnn(BaseCnnModel):

    def __init__(self,flags,mDB,gDB):
        super().__init__(flags)
        self.mDB = mDB
        self.DB = gDB
        #all_words = self.DB.select_top_k_words(['training_text','test_text_filter'],"Text",mode='tf',k=5)
        all_words = self.mDB.select_top_k_words(['training_text','test_text_filter'],"Text",mode='tfidf')
        #print(len(all_words),'also' in all_words,'gene' in all_words)
        #assert 0 
        self.DB.get_words(all_words)
        self.DB.y = self.mDB.y.copy()
        self.DB.get_clean_doc(['training_text','test_text_filter','stage2_test_text'],"Text",all_words)
        self.mDB.get_clean_doc(['training_text','test_text_filter','stage2_test_text'],"Text",all_words)
        self.V = len(all_words)

    def post(self):
        if self.flags.task == "test_cnn_stage1":
            docs = self.DB.clean_doc['test_text_filter']
        elif self.flags.task == "test_cnn_stage2":
            docs = self.DB.clean_doc['stage2_test_text']
        else:
            self.mDB.get_split()
            docs = self.mDB.split[self.flags.fold][1]
        nrows = len(docs)
        p = np.zeros([nrows,9])
        for i in range(self.flags.epochs):
            if i==0:
                skiprows=None
            else:
                skiprows = nrows*i
            p = p + (pd.read_csv(self.flags.pred_path,header=None,nrows=nrows,skiprows=skiprows).values)
        p = p/self.flags.epochs
        if '_cv' in self.flags.task:
            from utils.np_utils.utils import cross_entropy
            y = np.argmax(self.mDB.y,axis=1)
            print("cross entropy", cross_entropy(y[self.mDB.split[self.flags.fold][1]],p))
        s = pd.DataFrame(p,columns=['class%d'%i for i in range(1,10)])
        s['ID'] = np.arange(nrows)+1
        s.to_csv(self.flags.pred_path.replace(".csv","_sub.csv"),index=False,float_format="%.5f")

    def set_train_var(self):
        # This function could be overwritten
        self.var_list = [var for var in tf.trainable_variables() if "CBOW" not in var.name]

    def train(self):
        va = int(self.flags.fold>=0)
        self.train_from_placeholder(va=va) # this will call _build()

    def predict(self):
        #if os.path.exists(self.flags.pred_path)==0:
        self.predict_from_placeholder("softmax") # this will call _build() 
        self.post()

    def _build(self):
        V = self.V # vocabulary size
        M = self.flags.embedding_size # 64
        C = self.flags.classes
        W = self.flags.window_size
        S = self.flags.seq_len*2+1
        B = self.flags.batch_size
        H = 32
        is_training = tf.placeholder(dtype=tf.bool)
        netname = "CBOW"
        with tf.variable_scope(netname):
            self.inputs = tf.placeholder(dtype=tf.int32,shape=[None, S]) #[B,S]
            # each element is a word id.

            layer_name = "{}/embedding".format(netname)
            x = self._get_embedding(layer_name, self.inputs, V, M, reuse=False) # [B, S, M]       
        netname = "BaoBaoMiaoCnn"
        with tf.variable_scope(netname):
            x = tf.expand_dims(x, axis=3) # [B,S,M,1]
            
            net1 = self.conv_maxpool(x,W,M,S,H,"%s/conv1"%netname,1)     # [B,1,1,16]
            net2 = self.conv_maxpool(x,W*2,M,S,H,"%s/conv2"%netname,1)
            net3 = self.conv_maxpool(x,W//2,M,S,H,"%s/conv3"%netname,1)
            net = tf.concat([net1,net2,net3],axis=3) # [B,1,1,48]
            net = self._batch_normalization(net, layer_name='%s/batch_norm1'%(netname))
            net = tf.squeeze(net) # [B,48]
            #net = self._fc(net, fan_in=H*3, fan_out=H, layer_name="%s/fc0"%netname, activation='relu')
            net = self._fc(net, fan_in=H*3, fan_out=C, layer_name="%s/fc1"%netname, activation=None)
            self.logit = net
        self.is_training = is_training

    def conv_maxpool(self,net,W,M,S,N,name,sy):
        """
        Input: W: num_rows of filter
               M: num_cols of filter: embedding 
               S: seq_len
               N: num_filters
        """
        net = self._conv2D(net, [W,M], 1, N, strides=[1,sy,1,1], 
            layer_name="%s"%name, padding='VALID', activation="relu") #[B, S-W+1, 1, 16]
        net = tf.nn.max_pool(net,ksize = [1,(S-W)//sy+1,1,1], strides = [1,1,1,1], padding = 'VALID')
        return net

    def _batch_gen_test(self):
        from random import sample,randint
        self.DB.get_split()
        clean_doc = self.DB.clean_doc
        mclean_doc = self.mDB.clean_doc

        epochs = self.flags.epochs
        w2id = self.DB.w2id

        W = self.flags.seq_len
        if self.flags.task == "test_cnn_stage1":
            docs = clean_doc['test_text_filter']
            mdocs = mclean_doc['test_text_filter']
        elif self.flags.task == "test_cnn_stage2":
            docs = clean_doc['stage2_test_text']
            mdocs = mclean_doc['stage2_test_text']
        
        if self.flags.task == "test_cnn_cv":
            docs = clean_doc['training_text']
            mdocs = mclean_doc['training_text']
            fold = self.flags.fold
            docs_ids = list(self.DB.split[fold][1])

        else:
            docs_ids = list(range(len(docs)))
        
        docs_len = [len(doc) for doc in docs]

        from utils.draw.sns_draw import distribution
        distribution(docs_len)
        B = len(docs_ids)
        batches_per_epoch = 1
        print("Batch size:%d batches per epoch:%d"%(B, batches_per_epoch))
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                zeros = []
                for idx in docs_ids:
                    doc = docs[idx]
                    doc_len = len(doc)
                    while doc_len//4<W:
                        doc = mdocs[idx]+doc+mdocs[idx]
                        doc_len = len(doc)
                    
                    target_id = randint(int(doc_len*0.25),int(doc_len*0.75))
                    assert target_id-W>=0 and target_id+W+1<doc_len
                    pos = doc[target_id-W : target_id+W+1]
                    pos = [self.DB.w2id[word] for word in pos]
                    zeros.append(sum([1 for i in pos if i==0])*1.0/len(pos))
                    assert len(pos) == W*2+1
                    inputs.append(pos)
                #print("mean zeros",np.mean(zeros))
                yield inputs, None, epoch

    def _batch_gen_va(self):
        from random import sample,randint
        self.DB.get_split()
        clean_doc = self.DB.clean_doc
        mclean_doc = self.mDB.clean_doc

        epochs = 1
        w2id = self.DB.w2id

        W = self.flags.seq_len
        docs = clean_doc['training_text']
        mdocs = mclean_doc['training_text']

        fold = self.flags.fold
        docs_ids = list(self.DB.split[fold][1])

        docs_len = [len(doc) for doc in docs]
        #from utils.draw.sns_draw import distribution
        #distribution(docs_len)
        B = min(self.flags.batch_size,len(docs_ids))
        batches_per_epoch = 10

        y = self.DB.y
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                zeros = []
                idxs = sample(docs_ids,B)
                for idx in idxs:
                    doc = docs[idx]
                    doc_len = len(doc)
                    while doc_len//4<W:
                        doc = mdocs[idx]+doc+mdocs[idx]
                        doc_len = len(doc)
                    
                    target_id = randint(int(doc_len*0.25),int(doc_len*0.75))
                    assert target_id-W>=0 and target_id+W+1<doc_len
                    pos = doc[target_id-W : target_id+W+1]
                    pos = [self.DB.w2id[word] for word in pos]
                    zeros.append(sum([1 for i in pos if i==0])*1.0/len(pos))
                    assert len(pos) == W*2+1
                    inputs.append(pos)
                    labels.append(y[idx])
                #print("mean zeros",np.mean(zeros))
                
                yield inputs, labels, epoch


    def _batch_gen(self):
        from random import sample,randint
        self.DB.get_split()
        clean_doc = self.DB.clean_doc
        mclean_doc = self.mDB.clean_doc

        epochs = self.flags.epochs
        w2id = self.DB.w2id

        W = self.flags.seq_len
        
        docs = clean_doc['training_text']
        mdocs = mclean_doc['training_text']

        fold = self.flags.fold
        if fold>=0:
            docs_ids = list(self.DB.split[fold][0])
        else:
            docs_ids = list(range(len(docs)))


        docs_len = [len(doc) for doc in docs]
        #from utils.draw.sns_draw import distribution
        #distribution(docs_len)
        B = min(self.flags.batch_size,len(docs_ids))
        batches_per_epoch = 10#sum([l//B for l in docs_len])//20

        y = self.DB.y
        print("Batch size:%d batches per epoch:%d"%(B, batches_per_epoch))

        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                zeros = []
                xf = []
                idxs = sample(docs_ids,B)
                for idx in idxs:
                    doc = docs[idx]
                    doc_len = len(doc)
                    while doc_len//4<W:
                        doc = mdocs[idx]+doc+mdocs[idx]
                        doc_len = len(doc)
                    
                    target_id = randint(int(doc_len*0.25),int(doc_len*0.75))
                    assert target_id-W>=0 and target_id+W+1<doc_len
                    pos = doc[target_id-W : target_id+W+1]
                    pos = [self.DB.w2id[word] for word in pos]
                    zeros.append(sum([1 for i in pos if i==0])*1.0/len(pos))
                    assert len(pos) == W*2+1
                    inputs.append(pos)
                    labels.append(y[idx])
                #print("mean zeros",np.mean(zeros))
                yield inputs, labels, epoch
