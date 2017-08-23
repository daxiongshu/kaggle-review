import tensorflow as tf
from utils.tf_utils.BaseSeqData import BaseSeqData
from comps.instacart.basket_db import basketDB
from comps.instacart.sol43.insta_pb2 import User, Order
from collections import namedtuple
import os
import numpy as np
from utils.utils import print_mem_time
import pandas as pd
import gc

TEST_UID = 2455
LOG_EVERY = 100

class tfData(BaseSeqData):

    def __init__(self,flags):
        super().__init__(flags)
        self.pdDB = None
        self._write_user_tfrecord()

    def _write_user_tfrecord(self):
        outpath = "%s/users.tfrecords"%self.flags.record_path
        if os.path.exists(outpath)==True:
            print("%s exists."%outpath)
            return
        self._load_dic()
        inpath = self.flags.input_path
        self._load_db(files=["orders"])
        orders = self.pdDB.data["orders"].set_index('order_id',drop=0)      
        writer = tf.python_io.TFRecordWriter(outpath)
        i = 0
        for uid, oids in self.u2o.iteritems():
            user = User()
            user.uid = uid
            ordered_orders = orders.loc[oids].sort_values('order_number')
            for oid, orow in ordered_orders.iterrows():
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

    def _load_dic(self):
        path = self.flags.data_path
        p = "%s/u2o.pkl"%path
        if os.path.exists(p)==False:
            self._load_db()        
            u2o = self.pdDB.data['orders'].groupby('user_id')['order_id'].apply(list) 
            u2o.to_pickle(p)
        else:
            u2o = pd.read_pickle(p)
        print_mem_time("Loaded u2o %d"%len(u2o))

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
        self.u2o, self.o2p = u2o, o2p
        print_mem_time("Loaded o2p %d"%len(o2p))
        self._reset_pdDB()

    def _reset_pdDB(self):
        del self.pdDB
        gc.collect()
        self.pdDB = None

    def _load_db(self, files="all"):
        if self.pdDB is not None:
            return
        path = self.flags.input_path
        Table = namedtuple('Table', 'name fname dtype')
        fnames = "order_products__train.csv,order_products__prior.csv,orders.csv,aisles.csv,departments.csv".split(',')
        names = "op_train,op_prior,orders,aisles,departments".split(',')
        TABLES = [Table(i,"%s/%s"%(path,j),{}) for i,j in zip(names,fnames) if files =="all" or i in files]
        self.pdDB = basketDB(self.flags,TABLES,prob_dtype=True)


