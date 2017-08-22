from utils.tf_utils.BaseSeqData import BaseSeqData
from comps.instacart.basket_db import basketDB
from collections import namedtuple
import os
import numpy as np
from utils.utils import print_mem_time
import pandas as pd

class tfData(BaseSeqData):

    def __init__(self,flags):
        super().__init__(flags)
        self.pdDB = None
        self._load_dic()

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
        print_mem_time("Loaded o2p %d"%len(o2p))

    def _load_db(self):
        if self.pdDB is not None:
            return
        path = self.flags.input_path
        Table = namedtuple('Table', 'name fname dtype')


        TABLES = [Table('op_train',"%s/order_products__train.csv"%path,{}),
            Table('op_prior',"%s/order_products__prior.csv"%path,{}),
            Table('orders',"%s/orders.csv"%path,{}),
            Table('aisles',"%s/aisles.csv"%path,{}),
            Table('departments',"%s/departments.csv"%path,{}),
        ]
        self.pdDB = basketDB(self.flags,TABLES,prob_dtype=True)


    def _write_user_tfrecord(self):
        outpath = "%s/users.tfrecords"%self.flags.record_path
        if os.path.exists(outpath)==True:
            print("%s exists."%outpath)
            return 
        inpath = self.flags.input_path
        writer = tf.python_io.TFRecordWriter(outpath)
         
        writer.close()
