"""
xgboost for pairwise ranking
"""
import xgboost as xgb
import numpy as np

class xgb_rank(object):
    
    def __init__(self,params):
        self.params = params

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
                early_stopping_rounds=early_stopping_rounds,verbose_eval=1,xgb_model=load_model)
        else:
            watchlist = [(dtrain, 'train')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                verbose_eval=1,xgb_model=load_model)
        self.bst = bst
        if save_model is not None:
            bst.save_model(save_model)            
        

    def predict(self,Xt,Xg):
        dtest = xgb.DMatrix(Xt)
        dtest.set_group(Xg)
        return self.bst.predict(dtest)

    def feature_importance(self):
        return self.bst.get_fscore()
