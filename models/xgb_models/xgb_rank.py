"""
xgboost for pairwise ranking
"""
import xgboost as xgb
import numpy as np
from utils.pypy_utils.utils import sort_value
class xgb_rank(object):
    
    def __init__(self,params):
        self.params = params
        self.bst = None

    def fit(self,X,y,Xg,Xt=None,yt=None,Xgt=None,load_model=None,save_model=None):
        print(X.shape,y.shape)
        num_round = self.params['num_round']
        early_stopping_rounds = self.params['early_stopping_rounds']
        dtrain = xgb.DMatrix(X, y)
        dtrain.set_group(Xg)

        if Xt is not None:
            dvalid = xgb.DMatrix(Xt, yt)
            dvalid.set_group(Xgt)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                early_stopping_rounds=early_stopping_rounds,verbose_eval=1,xgb_model=load_model,
                maximize=True)
        else:
            watchlist = [(dtrain, 'train')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                verbose_eval=1,xgb_model=load_model)
        self.bst = bst
        if save_model is not None:
            bst.save_model(save_model)            
        

    def predict(self,Xt,Xg,load_model=None):
        print("load_model",load_model)
        dtest = xgb.DMatrix(Xt)
        dtest.set_group(Xg)
        if load_model and self.bst is None:
            self.bst = xgb.Booster(self.params,model_file=load_model)
        return self.bst.predict(dtest)


    def feature_importance(self):
        fscore = self.bst.get_fscore()
        feas = sort_value(fscore)
        return [(fea,fscore[fea]) for fea in feas]

