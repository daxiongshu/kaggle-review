from models.xgb_models.xgb_model import xgb_model 
from comps.personal.baobao.fe import build_feature,to4c
import numpy as np
import pandas as pd
from utils.np_utils.encoder import onehot_encode

if True:
    params = {"objective": "multi:softprob",
          "booster": "gbtree",
          "eval_metric": "mlogloss",
          "eta": 0.1,
          "max_depth": 5,
          #"min_child_weight":5,
          "silent": 1,
          "num_round": 1000,
          "subsample":0.5,
          "colsample_bytree":0.5,
          "early_stopping_rounds":10,
          "num_class":9,
          "tree_method":"exact"
          }


def cv(flags):
    X,y,Xt,yt,idx = build_feature(flags)
    
    params['verbose_eval'] = 10
    
    if '4c' in flags.task:
        y = np.argmax(to4c(onehot_encode(y)),axis=1)
        yt = np.argmax(to4c(onehot_encode(yt)),axis=1)
    params['num_class'] = np.max(y)+1
    model = xgb_model(params)
    print(X.shape,Xt.shape,y.shape,yt.shape)
    model.fit(X,y,Xt,yt,print_fscore=False)   
    yp = model.predict(Xt)
    s = pd.DataFrame(yp,columns=['class%d'%i for i in range(1,yp.shape[1]+1)])
    s['real'] = np.array(yt)
    s['ID'] = idx
    path = flags.data_path
    fold = flags.fold
    s.to_csv('%s/cv_%d.csv'%(path,fold),index=False)
    from utils.np_utils.utils import cross_entropy
    print(cross_entropy(yt,yp))


def sub(flags):
    X,y,Xt,_,_ = build_feature(flags)
    if '4c' in flags.task:
        y = np.argmax(to4c(onehot_encode(y)),axis=1)
    print(X.shape,Xt.shape,y.shape)
    params['num_class'] = np.max(y)+1
    params['num_round'] = 90 
    params["early_stopping_rounds"] = None
    params['verbose_eval'] = 100
    yp = np.zeros([Xt.shape[0],9])
    m = 5 if 'bag' in flags.task else 1
    for i in range(m):
        params['seed'] = i*9
        model = xgb_model(params)
        model.fit(X,y,print_fscore=False)
        tmp = model.predict(Xt)
        print(i,np.mean(tmp))
        yp += tmp
    yp/=m
    s = pd.DataFrame(yp,columns=["class%d"%i for i in range(1,yp.shape[1]+1)])
    s['ID'] = 1+np.arange(yp.shape[0])
    s.to_csv(flags.pred_path,index=False)

def post_cv(flags):
    import re
    import os
    path = flags.data_path
    files = [i for i in os.listdir(path) if len(re.findall('cv_[0-9].csv',i))]
    s = []
    for name in files:
        s.append(pd.read_csv("%s/%s"%(path,name)))
    
    s = pd.concat(s,axis=0)
    print(s.head())
    classes = len([i for i in s.columns.values if 'class' in i])
    from utils.np_utils.utils import cross_entropy
    yp = s[['class%d'%i for i in range(1,classes+1)]].values
    y=s['real'].values
    print(cross_entropy(y,yp))
    s.to_csv("%s/cv.csv"%path,index=False)

