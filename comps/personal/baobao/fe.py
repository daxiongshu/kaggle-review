# feature engineering
from comps.personal.personal_db import personalDB
from comps.personal.baobao.gene_text_db import geneTextDB
from comps.personal.baobao.write_gene import gene_map
from utils.draw.sns_draw import distribution
import numpy as np
from collections import defaultdict
from utils.pypy_utils.utils import sort_value
import pandas as pd
from utils.pd_utils.encoder import lbl_encode
import re
import os

def build_feature(flags,feas=None):

    if feas is None:
        feas = ['domain','full_text','gene_text','var_text','share','gene','pattern','onehot']
    if 'meta' in flags.task:
        feas.append('meta')
 
    myDB = personalDB(flags,name='full')
    myDB.get_per_sample_tfidf(['training_text','test_text_filter','stage2_test_text'],"Text")
    myDB.get_split()

    geneDB = geneTextDB(flags,W=10,bar=0)
    geneDB.poke()#_text()
    geneDB.get_per_sample_tfidf(['training_text','test_text_filter','stage2_test_text'],"Text")
  
    varDB = geneTextDB(flags,tag='variation',W=10,bar=0)
    varDB.poke()#_text()
    varDB.get_per_sample_tfidf(['training_text','test_text_filter','stage2_test_text'],"Text")

    fold,folds = flags.fold,flags.folds
    if fold>=0:
        tr_rows,te_rows = myDB.split[fold]
        tr_rows = tr_rows.tolist()
        te_rows = te_rows.tolist()
    else:
        tr_rows = None
        te_rows = 'stage2' if 'stage2' in flags.task else 'stage1'
    
    words = myDB.select_top_k_words(['training_text','test_text_filter'],"Text",mode="tf",k=5,slack=0)
    myDB.get_words(words)

    gwords = geneDB.select_top_k_words(['training_text','test_text_filter'],"Text",mode="tf",k=2,slack=0)
    geneDB.get_words(gwords)

    vwords = varDB.select_top_k_words(['training_text','test_text_filter'],"Text",mode="tf",k=5,slack=0)
    varDB.get_words(vwords)

    X,Xt = [],[]

    if 'meta' in feas:
        paths = [#"comps/personal/baobao/backup/91_88_87_86_74",
           # "comps/personal/baobao/backup/107_110_105_102_102",
           "comps/personal/baobao/backup/4c_52_51_47_46_39"]
        for path in paths:
            fill(X,Xt,*get_meta(myDB, tr_rows, te_rows, path))
    if 'text_len' in feas:
        fill(X,Xt,*get_text_len(myDB, tr_rows, te_rows))
    if 'domain' in feas:
        fill(X,Xt,*domain(myDB, tr_rows, te_rows))

    if 'full_text' in feas:
        for mode in ["tf"]:
            fill(X,Xt,*get_count(myDB, tr_rows, te_rows, mode))

    if 'gene_text' in feas:
        for mode in ["tf"]:
            fill(X,Xt,*get_count(geneDB, tr_rows, te_rows, mode))

    if 'var_text' in feas:
        for mode in ["tf"]:
            fill(X,Xt,*get_count(varDB, tr_rows, te_rows, mode))

    if 'share' in feas:
        fill(X,Xt,*get_share(myDB, tr_rows, te_rows))

    if 'pattern'  in feas:
        patterns = [r'[a-zA-Z][0-9]+[a-zA-Z]*','del']#,'ins','fus','trunc','methy','amp','sil','expr','splice','exon']
        fill(X,Xt,*get_pattern(myDB, tr_rows, te_rows,patterns))

    if 'gene' in feas:
        fill(X,Xt,*get_gene(myDB, tr_rows, te_rows))

    if 'd2v' in feas:
        fill(X,Xt,*get_d2v(flags.load_path, tr_rows, te_rows))

    if 'onehot' in feas:
        fill(X,Xt,*onehot_gene(myDB, tr_rows, te_rows))
        
    y,yt = myDB.gety(tr_rows,te_rows)    
    return np.hstack(X),y,np.hstack(Xt),yt,te_rows

def filter_words(words,patterns):
    result = set()
    for word in words:
        for pattern in patterns:
            if len(re.findall(pattern,word))>0:
                result.add(word)
                break
    return result

def get_text_len(DB, tr, te):
    if tr is None:
        if te=='stage1':
            Data = [DB.data['training_text'],DB.data['test_text_filter']]
        else:
            Data = [pd.concat([DB.data['training_text'],DB.data['test_text_filter']],axis=0),DB.data['stage2_test_text']]
    else:
        Data = [DB.data['training_text']]
    for data in Data:
        data['tl'] = data['Text'].apply(lambda x:len(x))
        data['tl2'] = data['Text'].apply(lambda x:len(x.split()))
    if tr is None:
        X,Xt = Data
        return X[['tl','tl2']].values, Xt[['tl','tl2']].values
    else:
        X = Data[0][['tl','tl2']].values
        return X[tr],X[te]

def get_meta(DB, tr, te,path):
    split = DB.split
    X = None
    for fold in range(4):
        name = "%s/cv_%d.csv"%(path,fold)
        if os.path.exists(name) == False:
            name = "%s/cnn_pred_cv%d_sub.csv"%(path,fold)
        s = pd.read_csv(name)
        clss = len([i for i in s.columns.values if 'class' in i])
        if X is None:
            X = np.zeros([DB.y.shape[0],clss])
        X[split[fold][1]] = s[['class%d'%i for i in range(1,clss+1)]].values
    if tr is None:
        name = "%s/cnn_pred_stage1_sub.csv"%(path)
        if os.path.exists(name) == False:
            name = "%s/sub_stage1.csv"%(path)
        Xt = pd.read_csv(name)
        clss = len([i for i in Xt.columns.values if 'class' in i])
        Xt = Xt[['class%d'%i for i in range(1,clss+1)]].values
        if te=='stage1':
            Data = [X,Xt]
        else:            
            X = np.hstack([X,Xt])
            name = "%s/cnn_pred_stage2_sub.csv"%(path)
            if os.path.exists(name) == False:
                name = "%s/sub_stage2.csv"%(path)
            Xt = pd.read_csv(name)
            clss = len([i for i in Xt.columns.values if 'class' in i])
            Xt = Xt[['class%d'%i for i in range(1,clss+1)]].values
            Data = [X,Xt]
    else:
        Data = [X]
    result = []
    for X in Data:
        if X.shape[1]==9:
            Xs = to4c(X)
        else:
            Xs = X
        result.append(Xs.argsort().argsort())
    if tr is None:
        return result[0],result[1]
    else:
        return result[0][tr],result[0][te]

def get_pattern(DB,tr,te,patterns):
    cols = ['p%d'%c for c,p in enumerate(patterns)]
    if tr is None:
        test = DB.data['test_variants_filter'] if te=='stage1' else DB.data['stage2_test_variants']
        if te=='stage1':
            train = DB.data['training_variants']
        else:
            train = pd.concat([DB.data['training_variants'],DB.data["test_variants_filter"]],axis=0)
        Data =[train,test]
    else:
        Data = [DB.data['training_variants']]

    for data in Data:
        for c,p in enumerate(patterns):
            data['p%d'%c] = data['Variation'].apply(lambda x: len(re.findall(p,str(x).lower())))

    if tr is None:
        return train[cols].values,test[cols].values
    else:
        X = data[cols].values
        return X[tr],X[te]

def domain(DB,tr,te):
    cols = ['gf','lf','tr']#,'ne','wt']
    words = [['gain of function','gain-of-function'],['loss of function','loss-of-function'],['transcript']]#,['neutral'],['wild type']]
    if tr is None:
        train = DB.data['training_text']
        if te=='stage1':
            test = DB.data['test_text_filter']
        else:
            train = pd.concat([DB.data['training_text'],DB.data["test_text_filter"]],axis=0)
            test = DB.data['stage2_test_text']
        Data = [train,test]
    else:
        Data = [DB.data['training_text']]

    for data in Data:
        for i,j in zip(cols,words):
            data[i] = 0
            for word in j:
                data[i] = data[i]+(data['Text'].apply(lambda x:len(re.findall(word, x.lower()))))
    if tr is None:
        print("domain",[train[i].mean() for i in cols])
        return train[cols].values,test[cols].values
    else:
        X = data[cols].values
        print("domain",[data[i].mean() for i in cols])
        return X[tr],X[te]

def get_gene(DB, tr, te):
    def _valid(x):
        s = re.findall(r'[a-zA-Z][0-9]+[a-zA-Z]*',x)
        return len(s)>0
    cols = "Variation,q1,q2,q3,q4".split(',')+['g%d'%i for i in range(8)]+['s%d'%i for i in range(8)]
    cols = cols + ['v%d'%i for i in range(8)]+['gv%d'%i for i in range(8)]
    if tr is None:
        train = DB.data['training_variants'] if te=='stage1' else pd.concat([DB.data['training_variants'],DB.data["test_variants_filter"]],axis=0)
        test = DB.data['test_variants_filter'] if te=='stage1' else DB.data['stage2_test_variants']
        Data = [train,test]
    else:
        Data = [DB.data['training_variants']]
    for data in Data:
        data['q1'] = data['Variation'].apply(lambda x: x.lower()[0] if _valid(x) else '')
        data['q2'] = data['Variation'].apply(lambda x: x.lower()[-1] if _valid(x) else '')
        data['q3'] = data['Variation'].apply(lambda x:len(x))
        data['q4'] = data['Gene'].apply(lambda x:len(x))
        #data['q5'] = data['Variation'].apply(lambda x: int(re.findall('[0-9]+',x)[0]) if _valid(x) else -1)
        for i in range(8):
            data['g%d'%i] = data['Gene'].apply(lambda x: x[i] if len(x)>i else '')
        for i in range(8):
            data['s%d'%i] = data['Variation'].apply(lambda x: x[i] if len(x)>i else '')
        for i in range(8):
            data['v%d'%i] = data['Variation'].apply(lambda x: x[:i+1] if len(x)>i else '')
        for i in range(8):
            data['gv%d'%i] = data.apply(lambda row: row['Gene']+'-'+row['Variation'][:i+1] if len(row['Variation'])>i else '',axis=1)
        #data['q5'] = data['Variation'].apply(lambda x:len(x.split()))
        #data['q6'] = data['Gene'].apply(lambda x:len(x.split()))
    if tr is None:
        lbl_encode(train,test,cols=cols)
        return train[cols].values,test[cols].values    
    else:
        lbl_encode(data,cols=cols)
        X = data[cols].values
        return X[tr],X[te]

def onehot_gene(DB, tr, te):
    from utils.np_utils.encoder import onehot_encode
    if tr is None:
        train = DB.data['training_variants']
        if te=="stage1":
            test = DB.data['test_variants_filter']
        else:
            train = pd.concat([train,DB.data['test_variants_filter']],axis=0)
            test = DB.data['stage2_test_variants']
        lbl_encode(train,test)
        n = max(train['Gene'].max(),test['Gene'].max())
        gtr = onehot_encode(train['Gene'].values,n=n+1)
        gte = onehot_encode(test['Gene'].values)
        return gtr,gte
    else:
        data = DB.data['training_variants']
        lbl_encode(data,cols=['Gene'])
        gene = data['Gene'].values
        gene = onehot_encode(gene)
        return gene[tr],gene[te]

def get_share(DB, tr, te):
    cols = "Gene,Variation".split(',')
    xcols = ["%s_%s"%(col,fea) for col in cols for fea in ['share','count','lcount']]
    
    if tr is None:
        train = pd.merge(DB.data['training_variants'],DB.data['training_text'],on='ID',how='left')
        if te=='stage2':
            tmp = pd.merge(DB.data['test_variants_filter'],DB.data['test_text_filter'],on='ID',how='left')
            train = pd.concat([train,tmp],axis=0)
            names = ['stage2_test_variants','stage2_test_text'] 
        else:
            names = ['test_variants_filter','test_text_filter']
        test = pd.concat([DB.data[names[0]],DB.data[names[1]]],axis=1)
        Data = [train,test]
    else:
        Data = [pd.concat([DB.data['training_variants'],DB.data['training_text']],axis=1)]

    for data in Data:
        for col in cols:
            data['%s_share'%col] = data.apply(lambda r: sum([1 for w in r[col].split(' ') if w.lower() in r['Text'].lower().split(' ')]), axis=1)
            if col=='Gene':
                data['%s_count'%col] = data.apply(lambda r: len(re.findall(gene_map(r[col]),r['Text'])), axis=1)
                data['%s_lcount'%col] = data.apply(lambda r: len(re.findall(gene_map(r[col]).lower(),r['Text'].lower())), axis=1)
            else:
                data['%s_count'%col] = data.apply(lambda r: sum([len(re.findall(i,r['Text'])) for i in r[col].split()]), axis=1)
                data['%s_lcount'%col] = data.apply(lambda r: sum([len(re.findall(i,r['Text'].lower())) for i in r[col].lower().split()]), axis=1)
    if tr is None:
        X,Xt = train[xcols].values,test[xcols].values
        return X,Xt
    else:
        X = data[xcols].values
        return X[tr],X[te]

def get_d2v(path,tr,te):
    d2v = np.load(path, encoding='latin1').item()['D2V/embedding/w:0']
    return d2v[tr],d2v[te]

def get_count(DB, tr_rows, te_rows, mode):
    if tr_rows is None:
        X = DB.get_list(mode,tr_rows,'training_text',"Text")
        if te_rows=='stage2':
            X.extend(DB.get_list(mode,tr_rows,'test_text_filter',"Text"))
        name = 'stage2_test_text' if te_rows=='stage2' else 'test_text_filter'
        Xt = DB.get_list(mode,te_rows,name,"Text")        
    else:
        X = DB.get_list(mode,tr_rows,'training_text',"Text")
        Xt = DB.get_list(mode,te_rows,'training_text',"Text") 
    return X,Xt


def topk_words_of_each_class(DB, k=10, mode="tfidf", use_folds=[],silent=0):
    DB.get_split()
    if mode == "tfidf":
        DB.get_per_sample_tfidf(['training_text'],"Text")
        docs = DB.sample_tfidf['training_text']
    else:
        assert 0
    y = DB.data["training_variants"]['Class']-1
    return _topk_words(DB,docs,y,k,mode,use_folds,silent)

def _topk_words(DB,docs,y,k,mode,use_folds,silent=0):
    names = 'Likely Loss-of-function,Likely Gain-of-function,Neutral,Loss-of-function,Likely Neutral,Inconclusive,Gain-of-function,Likely Switch-of-function,Switch-of-function'.split(',')
    count = {}
    docids = [i for fold in use_folds for i in DB.split[fold][1]]
    if silent == 0:
        print("Based on {}, Top {} words of each class:".format(mode,k))
    for cl in sorted(np.unique(y)):
        count[cl] = defaultdict(float)
    if silent == 0:
        print(docids[:10])
    for c in docids:
        doc = docs[c]
        cl = y[c]
        ss = sum(list(doc.values()))
        for word,score in doc.items():
            count[cl][word] += score/ss
    words = set()
    for cl in sorted(np.unique(y)):

        ws = sort_value(count[cl])[:k]
        #print(names[cl],cl)
        #print(ws)
        words = words.union(set(ws))
        if silent == 0:
            print("class {}, {}".format(cl,sort_value(count[cl])[:k]))
    return sorted(list(words))


def fill(X,Y,x,y):
    x,y = np.array(x),np.array(y)
    if x.shape[1]!=y.shape[1]:
        print(x.shape,y.shape)
        assert x.shape[1]==y.shape[1]
    X.append(x)
    Y.append(y) 

def to4c(X):
    Xs = np.zeros([X.shape[0],5])
    Xs[:,0] = X[:,0]+X[:,3]
    Xs[:,1] = X[:,1]+X[:,6]
    Xs[:,2] = X[:,2]+X[:,4]
    Xs[:,3] = X[:,7]+X[:,8]
    Xs[:,4] = X[:,5]
    return Xs

