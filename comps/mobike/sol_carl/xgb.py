from xgboost import XGBClassifier
from models.xgb_models.xgb_rank import xgb_rank
from utils.pypy_utils.utils import sort_value
import pandas as pd
import numpy as np
import gc
import os

def pre_data_chunk(trains,bad=[]):
    X,sub = [],None
    trains = [pd.read_csv(train, chunksize=3545320) for train in trains]
    bad = ['orderid','label','candidate_loc']+bad
    for dfs in zip(*trains):
        X = [df.drop([i for i in bad if i in df.columns.values],axis=1) for df in dfs]
        y = dfs[0]['label']
        sub = dfs[0][['orderid','candidate_loc']]
        group = dfs[0].groupby('orderid', sort = False).size()
        if len(X)>1:
            X = pd.concat(X,axis=1)#np.hstack(X)
        else:
            X = X[0]
        yield X,y,sub,group


def read_data(name):
    train_pk = name.replace('.csv','.pkl')
    if os.path.exists(train_pk) == False:
        train = pd.read_csv(name)
        if "va" not in name and "test" not in name:
            train.to_pickle(train_pk)
    else:
        train = pd.read_pickle(train_pk)
    return train

def pre_data(trains,istest,bad=[],mask=None):
    X,y,sub = [],None,None
    bad = ['orderid','label','candidate_loc']+bad
    for c,name in enumerate(trains):
        print(name)
        train = read_data(name)
        if c==0:
            if mask is not None:
                m = train['weekday'] == mask[0]
                if len(mask)>1:
                    for i in mask[1:]:
                        m = m | (train['weekday'] == i)
                train = train[m]
            if istest==0:
                y = train['label']
            else:
                sub = train[['orderid','candidate_loc']]
            group = train.groupby('orderid', sort = False).size()
        else:
            if mask is not None:
                train = train[m]
        badx = [ i for i in bad if i in train.columns.values]
        X.append(train.drop(badx,axis=1))
        del train
        gc.collect()
    if len(X)==1:
        return X[0],y,sub,group
    else:
        X = pd.concat(X,axis=1)#np.hstack(X)
        return X,y,sub,group

def train_predict(trains,samples,tests,out,obj="logloss",params={},load_model=None):
    run_xgb_rank(trains,samples,tests,out,params,load_model)


def run_xgb_rank(trains,samples,tests,out,params,load_model=None):
    model = xgb_rank(params)
    if load_model is None:
        if samples is not None:
            X,y,_,Xg = pre_data(trains,istest=0,mask=None)
            Xt,yt,_,Xgt = pre_data(samples,istest=0,mask=None)
            gc.collect()
            feas = X.shape[1]
            print(X.shape,y.shape,Xt.shape,yt.shape)
            model.fit(X,y,Xg,Xt,yt,Xgt,save_model="comps/mobike/sol_carl/data/xgb_cv.model")
            del X,y,Xt,yt,Xg,Xgt
        else:
            X,y,_,Xg = pre_data(trains,istest=0)
            model.fit(X,y,Xg,save_model="comps/mobike/sol_carl/data/xgb_sub.model")
            del X,y,Xg

    gc.collect()

    with open(out,'w') as f:
        f.write("orderid,candidate_loc,prob\n")
    try:
        for Xt,_,sub,Xg in pre_data_chunk(tests):
            sub['prob'] = model.predict(Xt,Xg,load_model)
            with open(out,'a') as f:
                sub.to_csv(f, header=False,index=False, float_format='%.5f')
    except:
        Xt,_,sub,Xg = pre_data(tests,istest=True)
        sub['prob'] = model.predict(Xt,Xg,load_model)
        with open(out,'a') as f:
            sub.to_csv(f, header=False,index=False, float_format='%.5f')
    fscore = model.feature_importance()
    print("Feature Importance:")
    for i in fscore:
        print(i) 


