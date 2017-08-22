"""
pandas database for data exploration and fast reloading
based on: https://github.com/colinmorris/instacart-basket-prediction/blob/master/preprocessing/basket_db.py
"""
import pandas as pd
import numpy as np
import os
import gc
try:
    import psutil
    _summ = psutil.virtual_memory()
except:
    pass

class pd_DB(object):
    def __init__(self,flags,tables=[],prob_dtype=False):
        """
            Input:
              tables: a list of namedtuples,
                which have attributes: name, fname, dtype
                name and fname are strings
                dtype is a {} columna name -> data type
                e.g. 'order_id': numpy.int32
                set tables to [] if only some member functions 
                are needed without loading data
              prob_dtype:
                if True, will detect optimal dtype automatically
                with additional time  
            build a self.data {} fname->pd data frame
        """
        print()
        self.flags = flags
        path = flags.data_path
        data = {}
        for table in tables:
            name = table.name
            fname = table.fname
            dtype = table.dtype
            pname = "%s/%s.pkl"%(path,name.split('/')[-1].split('.')[0])
            if os.path.exists(pname):
                data[name] = pd.read_pickle(pname)
            else:
                if len(dtype)==0 and prob_dtype:
                    dtype = self._get_dtype(fname)
                data[name] = pd.read_csv(fname,dtype=dtype)
                data[name].to_pickle(pname)
            try:
                print("Loaded",fname.split('/')[-1],data[name].shape, end='')
                summ = psutil.virtual_memory()
                print(" Used: %.2f GB %.2f%%"%((summ.used-_summ.used)/(1024.0*1024*1024),summ.percent-_summ.percent))
            except:
                print()              
        self.data = data # no copy, pass the inference
        print()

    def _get_dtype(self,fname):
        data = pd.read_csv(fname).fillna(0)
        cols = data.columns.values
        dtype = {}
        for col in cols:
            if data[col].dtype == np.float64:
                dtype[col] = np.float32
            elif data[col].dtype == np.int64:
                x = data[col].unique()
                mx = np.max(x)
                mi = np.min(x)
                if -128<mi and mx<127:
                    dtype[col] = np.int8
                elif 0<=mi and mx<256:
                    dtype[col] = np.uint8
                elif -32768<mi and mx< 32767:
                    dtype[col] = np.int16
                elif 0<=mi and mx<65535:
                    dtype[col] = np.uint16
                else:
                    dtype[col] = np.int32
            else:
                dtype[col] = data[col].dtype
        print("\n {} {} \n".format(fname,dtype))
        del data
        gc.collect()
        return dtype

    def snoop(self):
        raise NotImplementedError() 
