from utils.tf_utils.BaseSeqData import BaseSeqData
from comps.instacart.basket_db import basketDB
from collections import namedtuple
import os
import numpy as np


class tfData(BaseSeqData):

    def __init__(self,flags):
        super().__init__(flags)
        self._load_db()

    def _load_db(self):
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
