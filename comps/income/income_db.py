from collections import namedtuple,Counter,defaultdict
import os
import pandas as pd
import numpy as np
from utils.utils import print_mem_time
from utils.pd_utils.utils import series_equal,target_rate
from utils.pypy_utils.utils import load_pickle,save_pickle,sort_value
from utils.nlp_utils.nlp_pd_db import nlpDB
from utils.draw.sns_draw import distribution
import pickle

class incomeDB(nlpDB):

    def __init__(self,flags,name='full',files="all"):   
        super().__init__(name)
        self._build(flags,files)
        self.split = None

    def poke(self):
        #self.poke_target()
        #self.get_split()
        self.poke_num()
        #self.poke_user()

    def poke_num(self):
        train,test = self.data['train'],self.data['test']
        print(train.shape)
        for fea in train.columns.values:
            #if train[fea].dtype!='object':
                #print(train[fea].describe())
            print(fea, train[fea].unique().shape, type(train[fea].unique()))
            #print(target_rate(train,fea,'target'))
                #print()
        print(train['target'].unique())


    def _build(self,flags,files):
        path = flags.input_path
        Table = namedtuple('Table', 'name fname dtype')
        fnames = "adult.data,adult.test".split(',')
        names = "train,test".split(',')
        TABLES = [Table(i,"%s/%s"%(path,j),None) for i,j in zip(names,fnames) if files =="all" or i in files]

        print()
        self.flags = flags
        path = flags.data_path
        data = {}
        columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "gender",
            "capital_gain", "capital_loss", "hours_per_week", "native_country",
            "income_bracket"
        ]

        for table in TABLES:
            name = table.name
            fname = table.fname
            dtype = table.dtype
            pname = "%s/%s.pkl"%(path,name.split('/')[-1].split('.')[0])
            if os.path.exists(pname):
                data[name] = pd.read_pickle(pname)
            else:
                if name == 'train':
                    data[name] = pd.read_csv(fname,dtype=dtype,header=None,skipinitialspace=True,
                        names=columns)
                if name == 'test':
                    data[name] = pd.read_csv(fname,dtype=dtype,header=None,skipinitialspace=True,
                        skiprows=1,names=columns)
                data[name]['target'] = data[name]["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
                data[name].drop('income_bracket',axis=1,inplace=True)
                data[name].to_pickle(pname)
            print_mem_time("Loaded {} {}".format(fname.split('/')[-1],data[name].shape))
        self.data = data # no copy, pass the inference
        print()
    

