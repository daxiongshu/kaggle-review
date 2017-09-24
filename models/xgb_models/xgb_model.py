"""
xgboost for pairwise ranking
"""
import xgboost as xgb
import numpy as np
from utils.pypy_utils.utils import sort_value
 
class xgb_model(object):
    
    def __init__(self,params):
        self.params = params

    def fit(self,X,y,Xt=None,yt=None,
        load_model=None,save_model=None,
        obj=None,feval=None,print_fscore=True):
        print(X.shape,y.shape)

        num_round = self.params.get('num_round',100)
        early_stopping_rounds = self.params.get('early_stopping_rounds',None)
        dtrain = xgb.DMatrix(X, y)

        if Xt is not None:
            dvalid = xgb.DMatrix(Xt, yt)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                early_stopping_rounds=early_stopping_rounds,verbose_eval=1,
                xgb_model=load_model,obj=obj,feval=feval)
        else:
            watchlist = [(dtrain, 'train')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                verbose_eval=1,xgb_model=load_model,obj=obj,feval=feval)
        self.bst = bst
        if save_model is not None:
            bst.save_model(save_model)            
       
        fscore = self.feature_importance()
        if print_fscore:
            print("Feature Importance:")
            for i in fscore:
                print(i) 

    def predict(self,Xt):
        dtest = xgb.DMatrix(Xt)
        return self.bst.predict(dtest)

    def feature_importance(self):
        fscore = self.bst.get_fscore()
        feas = sort_value(fscore)
        return [(fea,fscore[fea]) for fea in feas]        
