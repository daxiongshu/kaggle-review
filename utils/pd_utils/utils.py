import pandas as pd
import numpy as np

def random_batch_gen(df,batch_size):
    B = batch_size
    frac = B*1.0/df.shape[0]
    print("run %d batches for 1 epoch"%(df.shape[0]//B))
    for j in range(df.shape[0]//B):
        x = df.sample(frac=frac).values
        yield x

def series_equal(s1,s2):
    return (s1==s2).all()

def sequential_iterate_df(df,batch_size):
    def _chunker(df, size):
        return (df[pos:pos + size] for pos in range(0, df.shape[0], size))
    for data in _chunker(df, batch_size):
        yield data

def target_rate(df,col,ycol,bar=10):
    vals = df[col].unique()
    if len(vals)>bar:
        return []
    xx = []
    for val in vals:
        mask = df[col] == val
        xx.append(df[mask][ycol].mean())
    return xx

def rm_categorical_cols(df,cols=None):
    print("rm categorical cols ...")
    bad = None
    if cols is None:
        cols = df.columns.values
    bad = [i for i in cols if df[i].dtype=='object']
    print("categorical cols {}".format(bad))
    df.drop(bad,axis=1,inplace=True)
    
def normalize(df,cols=None):
    print("normalize ...")
    if cols is None:
        cols = [i for i in df.columns.values if df[i].dtype!='object']
    df[cols] = (df[cols] - df[cols].mean())/df[cols].std()

def impute(df,cols=None,mode="mean"):
    print("impute %s ..."%mode)
    if mode == "mean":
        impute_mean(df,cols)
    else:
        print("unknown mode",mode)
        assert 0

def impute_mean(df,cols):
    if cols is None:
        cols = [i for i in df.columns.values if df[i].dtype!='object']
    df[cols] = df[cols].fillna(df[cols].mean())

def rm_const_cols(df,bar=0.999):
    print("remove const cols ...")
    cols = df.columns.values
    const = []
    for col in cols:
        tmp = df[col].value_counts()    
        ratio = tmp.max()/tmp.sum()
        if ratio>=bar:
            const.append(col)   
            print(ratio,col)
    print("df shape",df.shape,"num const cols",len(const))
    if len(const)>0:
        df.drop(const,axis=1,inplace=True)

def get_ymd(df,col,deli='-',order="ymd"):
    print("get year month day ...")
    def _parse_order(order):
        return {i:c for c,i in enumerate(order)}
    order_dic = _parse_order(order) 
    df["year"] = df[col].apply(lambda x: x.split(deli)[order_dic['y']]).astype(int)
    df["month"] = df[col].apply(lambda x: x.split(deli)[order_dic['m']]).astype(int)
    df["day"] = df[col].apply(lambda x: x.split(deli)[order_dic['d']]).astype(int)

def count_missing_per_row(df):
    print("count missinv values per row ...")
    df['num_missing'] = df.isnull().sum(axis=1)

#overfitting! try LOO!
def rank_cat(df_tr,ycol,df_te=None,cols=None,rank=True,tag=''):
    if cols is None:
        cols = [i for i in df_tr.columns.values if df_tr[i].dtype=='object']
    if len(cols)==0:
        print("no cat cols found")
        return
    for col in cols:
        dic = df_tr.groupby(col)[ycol].mean().to_dict()
        if rank:
            ks = [i for i in dic]
            vs = np.array([dic[i] for i in ks]).argsort().argsort()
            dic = {i:j for i,j in zip(ks,vs)}
        df_tr[tag+col] = df_tr[col].apply(lambda x: dic[x])
        if df_te is not None:
            df_te[tag+col] = df_te[col].apply(lambda x: dic.get(x,np.nan))

#overfitting! try LOO!
def std_cat(df_tr,ycol,df_te=None,cols=None,tag=''):
    if cols is None:
        cols = [i for i in df_tr.columns.values if df_tr[i].dtype=='object']
    if len(cols)==0:
        print("no cat cols found")
        return
    for col in cols:
        dic = df_tr.groupby(col)[ycol].std().to_dict()
        df_tr[tag+col] = df_tr[col].apply(lambda x: dic[x])
        if df_te is not None:
            df_te[tag+col] = df_te[col].apply(lambda x: dic.get(x,np.nan))    

def same_dtype_of_two_df(df1,df2):
    res = True
    for col in df1.columns.values:
        if col not in df2.columns.values:
            continue
        if df1[col].dtype!=df2[col].dtype:
            print(col,df1[col].dtype,df2[col].dtype)
            res = False
    print("yes" if res else "no")

def same_set_of_two_df(df1,df2):
    res = True
    for col in df1.columns.values:
        if df1[col].dtype!='object':
            continue
        if col not in df2.columns.values:
            continue
        s1 = set(df1[col].map(str).unique().tolist())
        s2 = set(df2[col].map(str).unique().tolist())
        if s1!=s2:
            if len(s1-s2)<10 and len(s2-s1)<10:               
                print(col,s1-s2,s2-s1)
            else:
                print(col,len(s1-s2),len(s2-s1))
            res = False
    print("yes" if res else "no")

def same_range_of_two_df(df1,df2):
    res = True
    for col in df1.columns.values:
        if df1[col].dtype=='object':
            continue
        if col not in df2.columns.values:
            continue
        if df1[col].min()!=df2[col].min() or df1[col].max()!=df2[col].max():
            print(col,[df1[col].min(),df1[col].max()],[df2[col].min(),df2[col].max()])
            res = False
    print("yes" if res else "no")
 
if __name__ == "__main__":
    s = pd.read_csv("../input/train.csv")
    rank_cat(s,'target',cols=['ps_ind_01'])
