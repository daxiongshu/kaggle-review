from utils.pd_utils.pd_db import pd_DB
from collections import namedtuple,Counter
import os
import pandas as pd
import numpy as np
from utils.utils import print_mem_time
import pickle
from utils.nlp_utils.utils import rm_stop_words,df_global_word_container,stem,\
    df_per_sample_word_lists,tf,idf,tf_idf,rm_punctuation
from utils.pypy_utils.utils import load_pickle,save_pickle,sort_value


class nlpDB(pd_DB):

    def __init__(self,name,noise_texts=[]):
        self.name = name
        self.stem_dic = None
        self.sample_tf = None
        self.sample_tfidf = None
        self.sample_words_count = None
        self.words_count = None
        self.global_idf_dic = None
        self.clean_doc = None
        self.noise_texts = noise_texts

    def get_list(self,name,rows,text,field):
        if name == "count" or name=="tf":
            self.get_per_sample_words_count([text],field,1)
            word_list = self.sample_words_count[text] 
            if name == "tf":
                word_list = tf(word_list)
        elif name == "tfidf":
            self.get_per_sample_tfidf([text],field,1)
            word_list = self.sample_tfidf[text]
        if isinstance(rows,list)==False:
            rows = list(range(len(word_list)))
        X = []
        num_words = len(self.w2id)
        for c in rows:
            count = word_list[c]
            X.append([count.get(self.id2w[i],0) for i in range(num_words)])
        return X

    def get_clean_doc(self, texts, field, selected_words):
        if self.clean_doc is not None:
            return

        self.clean_doc = {}

        name = "{}/{}_stem_dic.p".format(self.flags.data_path,self.name)
        self.stem_dic = load_pickle(self.stem_dic,name,{})
        assert len(self.stem_dic)

        for text in texts:
            name = "{}/{}_clean_doc_{}.p".format(self.flags.data_path,self.name,text)
            if os.path.exists(name):
                self.clean_doc[text] = pickle.load(open(name,'rb'))
            else:
                print("gen",name)
                word_lists = [] # list of lists, each item is a list of words for each sample
                df_per_sample_word_lists(self.data[text],field,word_lists) 
                # this function is in place.
                clean_lists = []
                for c,word_list in enumerate(word_lists):
                    word_list = rm_stop_words(word_list)
                    word_list = rm_punctuation(word_list)
                    word_list = stem(word_list,self.stem_dic)
                    word_list = [word for word in word_list if word in selected_words]
                    clean_lists.append(word_list)
                    if c%1000 == 0:
                        print("{} docs cleaned {}".format(c,word_list[:10]))
                pickle.dump(clean_lists,open(name,'wb'))
                self.clean_doc[text] = clean_lists
        #print(self.clean_doc[text][0])

    def get_per_sample_tfidf(self, texts, field, silent=0):
        """
        Each sample is a document.
        Input:
            texts: ["train","text"]
        """
        if self.sample_tfidf is not None:
            return

        self.sample_tfidf = {}
        self.get_per_sample_tf(texts, field, 1)

        name = "{}/{}_global_idf_dic.p".format(self.flags.data_path,self.name)
        self.global_idf_dic = load_pickle(self.global_idf_dic,name,{})
        if len(self.global_idf_dic)==0:
            print("gen",name)
            all_tf_list = []
            for text in texts:
                if text not in self.noise_texts:
                    all_tf_list.extend(self.sample_tf[text])
            idf(all_tf_list,self.global_idf_dic,0)
            save_pickle(self.global_idf_dic,name)

        for text in texts:
            name = "{}/{}_sample_tfidf_{}.p".format(self.flags.data_path,self.name,text)
            if os.path.exists(name):
                self.sample_tfidf[text] = pickle.load(open(name,'rb'))
            else:
                print("gen",name)
                tf_list = self.sample_tf[text]
                idf_list = self.get_idf_list(tf_list)
                tfidf_list = tf_idf(tf_list, idf_list,0)
                pickle.dump(tfidf_list,open(name,'wb'))
                self.sample_tfidf[text] = tfidf_list
            if silent==0:
                print("\n{} sample tfidf done".format(text))

    def get_idf_list(self,tf_list):
        idf_dic = self.global_idf_dic
        idf_list = []
        for tf_dic in tf_list:
            idf = {w:idf_dic.get(w,0) for w in tf_dic}
            idf_list.append(idf)
        return idf_list

    def get_per_sample_tf(self, texts, field, silent=0):
        """
        Each sample is a document.
        Input:
            texts: ["train","text"]
        """
        if self.sample_tf is not None:
            return

        self.sample_tf = {}
        self.get_per_sample_words_count(texts, field, 1)

        for text in texts:
            name = "{}/{}_sample_tf_{}.p".format(self.flags.data_path,self.name,text)
            if os.path.exists(name):
                self.sample_tf[text] = pickle.load(open(name,'rb'))
            else:
                print("gen",name)
                tf_list = tf(self.sample_words_count[text],0)
                pickle.dump(tf_list,open(name,'wb'))
                self.sample_tf[text] = tf_list
            if silent==0:
                print("\n{} sample tf done".format(text))


    def get_per_sample_words_count(self, texts, field, silent=0):
        """
        Each sample is a document.
        Input:
            texts: ["train","text"]
        """
        if self.sample_words_count is not None:
            return

        self.sample_words_count = {}
        self.get_global_words_count(texts,[field],1)

        for text in texts:
            name = "{}/{}_sample_count_{}.p".format(self.flags.data_path,self.name,text)
            if os.path.exists(name):
                self.sample_words_count[text] = pickle.load(open(name,'rb'))
            else:
                print("gen",name)
                word_lists = [] # list of lists, each item is a list of words for each sample
                df_per_sample_word_lists(self.data[text],field,word_lists) 
                # this function is in place.
                word_counts = []
                for word_list in word_lists:
                    word_list = rm_stop_words(word_list)
                    word_list = rm_punctuation(word_list)
                    word_list = stem(word_list,self.stem_dic)
                    word_counts.append(Counter(word_list))
                
                pickle.dump(word_counts,open(name,'wb'))
                self.sample_words_count[text] = word_counts
            if silent == 0:
                print("\n{} sample words count done".format(text))


    def get_global_words_count(self,texts,fields=["Text"],silent=0):
        """
        build self.words_count: {"train":Counter, "test":Counter}
        Input:
            texts: ["train","text"]
        """
        if self.words_count is not None:
            return
        
        self.words_count = {}
        name = "{}/{}_stem_dic.p".format(self.flags.data_path,self.name)
        self.stem_dic = load_pickle(self.stem_dic,name,{})

        for text in texts:
            name = "{}/{}_total_count_{}.p".format(self.flags.data_path,self.name,text)
            if os.path.exists(name):
            	self.words_count[text] = pickle.load(open(name,'rb'))
            else:
                print("gen",name)
                word_list = []
                df_global_word_container(self.data[text],fields,word_list) 
                # global word container means this is for the entire dataset, not per sample
                # this function is in place.

                word_list = rm_stop_words(word_list)
                word_list = rm_punctuation(word_list)
                word_list = stem(word_list,self.stem_dic)
                word_count = Counter(word_list)
                pickle.dump(word_count,open(name,'wb'))
                self.words_count[text] = word_count

            if silent==0:
                print("\nnumber of different words in {}:".format(text),len(self.words_count[text]))
                k = 10
                print("Top {} most common words in {}".format(k,text), self.words_count[text].most_common(k))

        name = "{}/{}_stem_dic.p".format(self.flags.data_path,self.name)
        save_pickle(self.stem_dic,name)

        self.global_word_count = Counter()
        for i,j in self.words_count.items():
            self.global_word_count = self.global_word_count + j

    def select_top_k_words(self, texts, field, mode="count", k=10, slack=8):
        name = "{}/{}_top{}-{}_{}_words.p".format(self.flags.data_path,self.name,k,slack,mode)
        selected = load_pickle(None,name,set())
        if len(selected):
            return selected
        print("gen",name)

        name = "{}/{}_stem_dic.p".format(self.flags.data_path,self.name)
        self.stem_dic = load_pickle(self.stem_dic,name,{})
        assert len(self.stem_dic)

        if mode=="count":
            self.get_per_sample_words_count(texts,field)
        elif mode == "tfidf":
            self.get_per_sample_tfidf(texts,field)
        elif mode=="tf":
            self.get_per_sample_tfidf(texts,field)
        else:
            print("unknown mode",mode)
            assert 0

        for text in texts:
            if mode=="count":
                data = self.sample_words_count[text]
            elif mode=="tfidf":
                data = self.sample_tfidf[text]
            elif mode=="tf":
                data = self.sample_tf[text]

            for c,wd in enumerate(data):
                topk = sort_value(wd)[:k+slack]
                topk = set([i for i in topk if len(i)>2])
                selected = selected.union(topk)
                if c>0 and c%1000 == 0:
                    print("{} documents done, mode {}, sample {}, num {}".format(c,mode,topk,len(selected)))
        print("num of selected {} key words:".format(mode),len(selected))
        name = "{}/{}_top{}-{}_{}_words.p".format(self.flags.data_path,self.name,k,slack,mode)
        save_pickle(selected,name)
        return selected

    def get_words(self,words):
        words = ["__NULL__"] + sorted(list(words))
        self.w2id = {i:c for c,i in enumerate(words)}
        self.id2w = {j:i for i,j in self.w2id.items()}


