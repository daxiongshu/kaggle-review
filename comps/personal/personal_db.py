from utils.nlp_utils.nlp_pd_db import nlpDB
from collections import namedtuple,Counter
import os
import pandas as pd
import numpy as np
from utils.utils import print_mem_time
from utils.pd_utils.utils import series_equal
from utils.pypy_utils.utils import load_pickle,save_pickle
from utils.nlp_utils.utils import df_per_sample_word_lists
import pickle

class personalDB(nlpDB):

    def __init__(self,flags,name,files="all",build=True,noise_texts=None):   
        if noise_texts is None:
            noise_texts=['stage2_test_text']
        super().__init__(name=name,noise_texts=noise_texts)
        self.fnames = "training_text,test_text_filter,test_variants_filter,training_variants,stage2_test_text.csv,stage2_test_variants.csv,stage2_sample_submission.csv".split(',')
        self.names = [i.split('.')[0] for i in self.fnames]
        self.split = None
        self.path = flags.input_path
        if build:
            self._build(flags,files)

    def gety(self,tr,te):
        y = self.data["training_variants"]['Class']-1
        if tr is None:
            if te=='stage1':
                return y,None
            else:
                y = y.append(self.data["test_variants_filter"]['Class']-1)
                return y,None
        else:
            return y[tr],y[te]

    def _build(self,flags,files):
        fnames,names = self.fnames,self.names
        path = self.path 
        Table = namedtuple('Table', 'name fname dtype')
        tables = [Table(i,"%s/%s"%(path,j),{}) for i,j in zip(names,fnames) if files =="all" or i in files]

        print()
        self.flags = flags
        path = flags.data_path
        data = {}
        for table in tables:
            name,fname,dtype = table.name,table.fname,table.dtype
            pname = "%s/%s_%s.pkl"%(path,self.name,name.split('/')[-1].split('.')[0])
            if os.path.exists(pname):
                data[name] = pd.read_pickle(pname)
            else: 
                if '_text' in name:              
                    data[name] = pd.read_csv(fname,header=None,sep="\|\|",skiprows=1,names=['ID','Text']) 
                else:
                    data[name] = pd.read_csv(fname)
                data[name].to_pickle(pname)
            print_mem_time("Loaded {} {}".format(fname.split('/')[-1],data[name].shape))
        self.data = data # no copy, pass the reference
        if "training_variants" in self.data:
            y = self.data["training_variants"]['Class']-1
            from utils.np_utils.encoder import onehot_encode
            self.y = onehot_encode(y,self.flags.classes)
        print()

    def poke_text(self):
        for name,data in self.data.items():
            if 'text' not in name:
                continue
            data['Text'] = data['Text'].apply(lambda x: str(x).encode('utf-8'))#.encode('utf-8')
            #print(name,data.head())
            data['lw'] = data['Text'].apply(lambda x: len(x.strip().split()))
            print(name,data.shape,data['lw'].mean()) 

    def poke(self):            
        #print("class distribuion")
        #print(self.data['training_variants']['Class'].value_counts())
        #self.poke_tfidf()
        return

        data = self.data['training_text']
        data['Gene'] = self.data['training_variants']['Gene'].values
        data['gin'] = data.apply(lambda r: r['Gene'].lower() in r['Text'].lower().split(), axis=1)
        mask = data['gin']==0
        print(data['gin'].mean())
        print(data[mask]['Gene'].unique())
        print(data[mask]['Gene'].unique().shape)
        print(data[mask]['ID'][:10])

    def poke_tfidf(self):
        self.get_per_sample_tfidf(['training_text','test_text_filter','stage2_test_text'],"Text")
        #self.random_pick_sample()
        words = self.select_top_k_tfidf_words(['training_text','test_text_filter'],"Text")
        self.get_split()
        for tr,te in self.split:
            print(tr.shape,te.shape,tr[-5:],te[-5:])

    def get_split(self):
        if self.split is not None:
            return
        name = "{}/split.p".format(self.flags.data_path)
        split = load_pickle(None,name,[])
        
        if len(split) == 0:
            #data = self.data["training_variants"].append(self.data["test_variants_filter"])
            data = self.data["training_variants"]
            y = data['Class']-1
            X = np.arange(y.shape[0])
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=self.flags.folds,shuffle=True,random_state=99)
            split = [(train_index, test_index) for train_index, test_index in skf.split(X, y)]
            save_pickle(split,name)
            print("new shuffle")
        self.split = split
        #print("split va",split[0][1][:10])

    def random_pick_sample(self, bar=0.005,floor = 1e-4):
        print("\n random pick sample with tfidf > {}:".format(bar))
        from random import randint
        data = self.sample_tfidf["training_text"]
        k = randint(0,len(data)-1)
        print([(word,tfidf) for word,tfidf in data[k].items() if tfidf>bar])

        print("\n random pick sample with tfidf < {}:".format(floor))
        k = randint(0,len(data)-1)
        print([(word,tfidf) for word,tfidf in data[k].items() if tfidf<floor])

    






    

    

