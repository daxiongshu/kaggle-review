import pandas as pd
from collections import Counter
from math import log



def tf_idf(tf_dic_list, idf_dic_list, silent=1):
    if silent == 0:
        print("tf idf ...")
    tf_idf_dic_list = [] # [{word:tf_idf} for each sample]
    for tf_dic,idf_dic in zip(tf_dic_list,idf_dic_list):
        tf_idf_dic = {i:tf_dic[i]*idf_dic[i] for i in tf_dic}
        tf_idf_dic_list.append(tf_idf_dic)
    return tf_idf_dic_list   

def tf(counter_list,silent=1):
    if silent==0:
        print("tf ...")
    tf_dic_list = [] # [{word:tf} for each sample]
    for counter in counter_list:
        count_sum = sum(counter.values())
        tf_dic = {word:counter[word]*1.0/count_sum for word in counter}
        tf_dic_list.append(tf_dic)
    return tf_dic_list 

def idf(tf_dic_list,global_idf_dic,silent=1):
    """
    Input:
        global_idf_dic = {} # word -> idf, which may be updated in place
    """
    if silent==0:
        print("idf ...")
    doc_len = len(tf_dic_list)
    idf_dic_list = [] # [{word:idf} for each sample]    

    for c,tf_dic in enumerate(tf_dic_list):
        idf_dic = {}
        for word in tf_dic:
            if word not in global_idf_dic:
                n_containing = sum([word in tf_dic for tf_dic in tf_dic_list])
                global_idf_dic[word] = log(doc_len/(1.0+n_containing))
            idf_dic[word] = global_idf_dic[word]
        idf_dic_list.append(idf_dic)
        if silent == 0 and c>0 and c%100 == 0:
            print("{} documents done, total {}, word {}, idf {}".format(c,len(tf_dic_list),word,global_idf_dic[word]))
    return idf_dic_list

def df_per_sample_word_lists(df,col,words):    
    df[col].str.lower().str.split().apply(words.append)
        

def df_global_word_container(df,cols,words):
    for col in cols:
        if isinstance(words,list):
            df[col].str.lower().str.split().apply(words.extend)
        else:
            df[col].str.lower().str.split().apply(words.update)

def stem(words,stem_dic,mode="nltk",silent=1):
    if silent==0:
        print("stem ...")
    if mode == "nltk":
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
    else:
        print("unknown mode",mode)
        assert 0
    for word in set(words):
        if word not in stem_dic:
            stem_dic[word] = stemmer.stem(word)
    words = [stem_dic[word] for word in words]
    return words

def rm_punctuation(data,pattern=r'[a-zA-Z]+-?[0-9]*',silent=1):
    if silent==0:
        print("remove punctuation ...")
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(pattern)
    return tokenizer.tokenize(" ".join(data))

def rm_stop_words(data, mode="nltk",silent=1):
    """
    Input:
        data is a set, {} or Counter
    """
    if silent==0:
        print("remove stop words ...")
    if mode == "nltk":
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    else:
        print("unknown mode",mode)
        assert 0

    if isinstance(data,list):   
        data = [i for i in data if i.lower() not in stop_words]
        return data
    else:
        for word in stop_words:
            if word in data:
                del data[word]
