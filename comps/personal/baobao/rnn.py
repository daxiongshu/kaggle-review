import tensorflow as tf
from comps.personal.personal_db import personalDB
from comps.personal.baobao.embedding import BaoEmbedding

def train_rnn(flags):
    myDB = personalDB(flags)
    model = BaoRnn(flags,myDB)
    model.train()


class BaoRnn(BaoEmbedding):

    def train(self):
        self.train_from_placeholder(1) # this will call _build()

    def set_train_var(self):
        # This function could be overwritten
        self.var_list = [var for var in tf.trainable_variables() if "CBOW" not in var.name]

    def _build(self):
        V = self.V
        M = self.flags.embedding_size # 64
        H = self.flags.num_units 
        C = self.flags.classes

        netname = "CBOW"
        with tf.variable_scope(netname):
            self.inputs = tf.placeholder(dtype=tf.int32,shape=[None, None]) #[B,S]
            layer_name = "{}/embedding".format(netname)
            x = self._get_embedding(layer_name, self.inputs, V, M, reuse=False) # [B, S, M]

        netname = "RNN"
        cell_name = self.flags.cell
        with tf.variable_scope(netname):
            args = {"num_units":H,"num_proj":C}
            cell_f = self._get_rnn_cell(cell_name=cell_name, args=args)
            cell_b = self._get_rnn_cell(cell_name=cell_name, args=args)
            (out_f, out_b), _ = tf.nn.bidirectional_dynamic_rnn(cell_f,cell_b,x,dtype=tf.float32)
            #logit = (out_f[:,-1,:] + out_b[:,-1,:])*0.5  # [B,1,C]
            logit = tf.reduce_mean(out_f+out_b,axis=1)
            logit = tf.squeeze(logit) # [B,C]
            self.logit = logit

    def _batch_gen_va(self):
        return self._batch_gen(1)

    def _batch_gen(self, va=0):
        from random import sample,randint
        self.DB.get_split()
        all_words = list(self.all_words)
        clean_doc = self.DB.clean_doc

        epochs = self.flags.epochs if va==0 else 1
        w2id = self.DB.w2id

        S = self.flags.seq_len
        V = self.V # vocabulary size, should be passed from DB

        docs = clean_doc['training_text']
        fold = self.flags.fold
        docs_ids = list(self.DB.split[fold][va])
        docs_len = [len(doc) for doc in docs]
        B = min(self.flags.batch_size,len(docs_ids))
        batches_per_epoch = sum([l//B for l in docs_len])

        y = self.DB.y
        #print(batches_per_epoch)
        for epoch in range(epochs):
            for batch in range(batches_per_epoch):
                inputs = []
                labels = [] # 0 or 1
                W = randint(int(S*0.5),int(S*1.5))
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
                    assert len(pos) == W*2+1
                    inputs.append(pos)
                    labels.append(y[idx])
                yield inputs, labels, epoch  
                if va and batch>10:
                    break
