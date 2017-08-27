import tensorflow as tf
from utils.tf_utils.BaseSeqData import BaseSeqData
from comps.instacart.basket_db import basketDB
from comps.instacart.sol43.insta_pb2 import User, Order
from comps.instacart.sol43.user_wrapper import UserWrapper 
from collections import namedtuple
from utils.tf_utils.utils import intseqfea,floatseqfea,_int64_feature,_float_feature 
import os
import numpy as np
from utils.utils import print_mem_time
import pandas as pd
import gc
import random 

TEST_UID = 2455 # randomly select one user in train for sanity check
LOG_EVERY = 100

class tfData(BaseSeqData):

    def __init__(self,flags):
        super().__init__(flags)
        self._setup()
        self._load_db()
        self._write_user_tfrecord()
        self._partition_test_user_tfrecord()
        self._write_train_tfrecord(max_prods=np.inf)
        #self._poke_seq_data()
        #self._poke_user_tfrecords()
  
    def _setup(self):
        self.pdDB = None
        self.o2p = None
        self.u2o = None
        self.p2adn = None 
        self.context_fields = [
            'pid', 'aisleid', 'deptid', 'uid', 'weight',
        ]
        self.raw_feats = ['previously_ordered',]
        self.generic_raw_feats = ['days_since_prior', 'dow', 'hour',
            'n_prev_products',
            'n_prev_repeats', 'n_prev_reorders'
        ] # order level features
        """
        n_prev_products: num of products in order i-1
        n_prev_reorders: num of reordered products in order i-1
        n_prev_repeats: num of products in order i-1 that is also in order i-2
        """
        self.sequence_fields = ['lossmask', 'labels', ]
        self.float_fields = "weight,labels,lossmask".split(',') # all other fields are integer.


    def _partition_test_user_tfrecord(self):
        outpath = "%s/test_users.tfrecords"%self.flags.record_path
        recordpath = "%s/users.tfrecords"%self.flags.record_path
        if os.path.exists(outpath)==True:
            print("%s exists."%outpath)
            return
        print("write",outpath)
        writer = tf.python_io.TFRecordWriter(outpath)
        records = tf.python_io.tf_record_iterator(recordpath)
        cc = 0
        for c,record in enumerate(records):
            user = User()
            user.ParseFromString(record)
            if user.test:
                writer.write(record)
                cc+=1
            if c%1000 == 0:
                print(c,cc)
        print("write %s done"%outpath)
        writer.close()
        

    def _write_train_tfrecord(self,max_prods):
        outpath = "%s/train.tfrecords"%self.flags.record_path
        if os.path.exists(outpath)==True:
            print("%s exists."%outpath)
            return
        path = "%s/users.tfrecords"%self.flags.record_path
        ctype = getattr(tf.python_io.TFRecordCompressionType, "GZIP") 
        writer_options = tf.python_io.TFRecordOptions(compression_type=ctype)
        writer = tf.python_io.TFRecordWriter(outpath, options=writer_options)
        ces = 0
        for cu,user in enumerate(self._iterate_wrapped_users(path, mode="train")):
            for ce,example in enumerate(self._get_user_sequence_examples(user, max_prods = max_prods)):
                writer.write(example.SerializeToString())
            ces += ce
            if cu>0 and cu%100 == 0:
                print_mem_time("%d users %d samples"%(cu,ces))

    def _get_user_sequence_examples(self,user,max_prods=np.inf,mode="train"):
        self._load_p2adn()
        prod_lookup = self.p2adn
        nprods = min(max_prods, user.nprods)
        weight = 1 / nprods # smaller weight for a user who purchase a lot
        # Generic context features
        base_context = {
            'uid': _int64_feature(user.uid),
            'weight': _float_feature(weight),
        }
        if nprods == user.nprods:
            pids = sorted(user.all_pids)
        else:
            pids = random.sample(user.all_pids, nprods)

        gfeats, pidfeats = self._build_seq_data(user, pids)
        base_seqdict = dict([(i,floatseqfea(j)) if i in self.float_fields else (i,intseqfea(j)) for i,j in gfeats.items()])
        for pidx, pid in enumerate(pids):
            # Context features (those that don't scale with seqlen)
            ctx_dict = base_context.copy()
            aisleid, deptid = prod_lookup.loc[pid,'aisle_id'],prod_lookup.loc[pid,'department_id']
            product_ctx = dict(pid=_int64_feature(pid), 
                aisleid=_int64_feature(aisleid), deptid=_int64_feature(deptid))
            ctx_dict.update(product_ctx)
            context = tf.train.Features(feature=ctx_dict)
            # Sequence features
            seqdict = base_seqdict.copy()
            product_seqdict = dict([(i,floatseqfea(j[pidx])) if i in self.float_fields else(i,intseqfea(j[pidx])) for i,j in pidfeats.items()])
            #product_seqdict = dict(to_seq_feat( (name, featarray[pidx]) )
            #    for name, featarray in prodfeats.iteritems())
            seqdict.update(product_seqdict)
            feature_lists = tf.train.FeatureLists(feature_list = seqdict)
            example = tf.train.SequenceExample(
                context = context,
                feature_lists = feature_lists,
            )
            yield example   
        

    def _build_seq_data(self, user, pids):
        """Return a tuple of (generic, product-specific) dicts
        The former's values have shape (seqlen), the latter has shape (npids, seqlen)
        pids could be a sample of user.all_pids
        """
        generic_raw_feats = self.generic_raw_feats
        # not specific to a product
        gfs = {featname: np.empty(user.seqlen) for featname in generic_raw_feats}
        nprods = len(pids)
        pid_to_ix = dict(zip(pids, range(nprods)))
        pidfeat_shape = (nprods, user.seqlen)
        labels = np.zeros(pidfeat_shape)
        prev_ordered = np.zeros(pidfeat_shape)
        pids_seen = set([]) # unique pids seen up to but not including the ith order
        prev_pidset = None
        for i, order in enumerate(user.orders):
            # The order with index i corresponds to the i-1th element of the sequence
            # (we always skip the first order, because by definition it can have no
            # reorders)
            # i = 0,1,...seqlen
            seqidx = i-1
            ordered = order.products
            unordered = set(ordered) # I made a funny
            # Calculate the generic (non-product-specific) features
            # Sometimes the value of the next order's feature is a function of this order
            if i < user.seqlen: # 0,1,..,seqlen-1
                gfs['n_prev_products'][i] = len(ordered)
                gfs['n_prev_reorders'][i] = len(pids_seen.intersection(unordered)) 
                gfs['n_prev_repeats'][i] = 0 if prev_pidset is None else len(prev_pidset.intersection(unordered))
            # And some features are calculated wrt the current order
            if i > 0: # 1,2,...,seqlen
                gfs['days_since_prior'][seqidx] = order.days_since_prior
                gfs['dow'][seqidx] = order.dow
                gfs['hour'][seqidx] = order.hour
            # Product specific feats
            for (cart_index, pid) in enumerate(ordered):
                try:
                    j = pid_to_ix[pid] #global pid_ix of all orders
                except KeyError:
                    continue
                if i < user.seqlen:
                    prev_ordered[j, i] = cart_index+1
                if i > 0:
                    labels[j, seqidx] = 1
            pids_seen.update(unordered)
            prev_pidset = unordered

        lossmask = np.zeros(pidfeat_shape)
        first_occurences = (prev_ordered>0).argmax(axis=1)
        # There's probably a clever non-loopy way to do this.
        for j, first in enumerate(first_occurences):
            lossmask[j, first:] = 1
        pidfeats = dict(labels=labels, lossmask=lossmask, previously_ordered=prev_ordered)
        return gfs, pidfeats
     
        

    def _poke_seq_data(self):
        path = "%s/users.tfrecords"%self.flags.record_path
        for user in self._iterate_wrapped_users(path, mode="train"):
            pids = sorted(user.all_pids) 
            gf,pf = self._build_seq_data(user, pids)
            print(pf['lossmask'],'\n')
            print(pids,'\n')
            print(pf['previously_ordered'],'\n')
            for order in user.orders:
                print(list(order.products)) 
            break

    def _poke_user_tfrecords(self):
        path = "%s/users.tfrecords"%self.flags.record_path
        for user in self._iterate_wrapped_users(path, mode="test"):
            #print(type(user))
            print(user.uid,user.istest)
            #break

    def _iterate_wrapped_users(self, recordpath, mode="train"):
        records = tf.python_io.tf_record_iterator(recordpath)
        for record in records:
            user = User()
            user.ParseFromString(record)
            yield UserWrapper(user, mode)    

    def _write_user_tfrecord(self):
        outpath = "%s/users.tfrecords"%self.flags.record_path
        if os.path.exists(outpath)==True:
            print("%s exists."%outpath)
            return
        self._load_u2o() # get u2o, o2p and p2adn
        self._load_o2p()
        inpath = self.flags.input_path
        self._load_db(files=["orders"])
        orders = self.pdDB.data["orders"].set_index('order_id',drop=0)      
        writer = tf.python_io.TFRecordWriter(outpath)
        i = 0
        for uid, oids in self.u2o.iteritems():
            user = User()
            user.uid = uid
            ordered_orders = orders.loc[oids].sort_values('order_number')
            for oid, orow in ordered_orders.iterrows(): # don't forget set_index!
                test = orow.eval_set == 'test'
                #print(oid,orow)
                if test:
                    user.test = True
                    order = user.testorder
                else:
                    order = user.orders.add() # what does this mean?
                order.orderid = oid
                order.nth = orow.order_number
                order.dow = orow.order_dow
                order.hour = orow.order_hour_of_day
                days = orow.days_since_prior_order
                if not pd.isnull(days):
                    order.days_since_prior = int(days)
                # If this is a test order, products gets left empty. We don't
                # know which products are in this order.
                if test:
                    #user.testorder = order
                    pass
                else:
                    order.products.extend(self.o2p.loc[oid])
            writer.write(user.SerializeToString())
            if uid == TEST_UID:
                print ("Writing uid {} to testuser.pb".format(uid))
                with open('%s/testuser.pb'%self.flags.record_path, 'wb') as f:
                    f.write(user.SerializeToString())
            i += 1
            if i % LOG_EVERY == 0:
                print_mem_time ("{} users written".format(i))
        writer.close()

    def _load_u2o(self):
        if self.u2o:
            return
        path = self.flags.data_path
        p = "%s/u2o.pkl"%path
        if os.path.exists(p)==False:
            self._load_db()        
            u2o = self.pdDB.data['orders'].groupby('user_id')['order_id'].apply(list) 
            u2o.to_pickle(p)
        else:
            u2o = pd.read_pickle(p)
        self.u2o = u2o
        print_mem_time("Loaded u2o %d"%len(u2o))

    def _load_o2p(self):
        if self.o2p:
            return
        path = self.flags.data_path
        p = "%s/o2p.pkl"%path
        if os.path.exists(p)==False:
            self._load_db()
            ops = self.pdDB.data['op_prior']
            ops = ops.append(self.pdDB.data['op_train'])
            o2p = ops.sort_values(['order_id', 'add_to_cart_order'])\
                .groupby('order_id')['product_id'].apply(list)
            o2p.to_pickle(p)
        else:
            o2p = pd.read_pickle(p)
        self.o2p = o2p
        print_mem_time("Loaded o2p %d"%len(o2p))


    def _load_p2adn(self):
        if self.p2adn is not None:
            return
        self._load_db()
        p2adn = self.pdDB.data['products']
        self.p2adn = p2adn.set_index('product_id')
        print_mem_time("Loaded p2adn %d"%len(p2adn))
        #self._reset_pdDB()

    def _reset_pdDB(self):
        del self.pdDB
        gc.collect()
        self.pdDB = None

    def _load_db(self, files="all"):
        if self.pdDB is not None:
            return
        path = self.flags.input_path
        Table = namedtuple('Table', 'name fname dtype')
        fnames = "order_products__train.csv,order_products__prior.csv,orders.csv,aisles.csv,departments.csv,products.csv".split(',')
        names = "op_train,op_prior,orders,aisles,departments,products".split(',')
        TABLES = [Table(i,"%s/%s"%(path,j),{}) for i,j in zip(names,fnames) if files =="all" or i in files]
        self.pdDB = basketDB(self.flags,TABLES,prob_dtype=True)


