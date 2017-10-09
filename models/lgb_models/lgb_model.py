import lightgbm as lgb 
import numpy as np
from utils.pypy_utils.utils import sort_value
 
class lgb_model(object):
    
    def __init__(self,params):
        self.params = params
        self.bst = None

    def fit(self,X,y,Xt=None,yt=None,
        load_model=None,save_model=None,
        obj=None,feval=None,print_fscore=True):
        print(X.shape,y.shape)

        num_round = self.params.get('num_round',100)
        early_stopping_rounds = self.params.get('early_stopping_rounds',None)
        dtrain = lgb.Dataset(X, y)

        if Xt is not None:
            dvalid = lgb.Dataset(Xt, yt, reference=dtrain)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = lgb.train(self.params,dtrain,
                num_boost_round=num_round,
                valid_sets=dvalid,
                early_stopping_rounds=early_stopping_rounds)
        else:
            bst = lgb.train(self.params,dtrain,
                num_boost_round=num_round)
        self.bst = bst
        if save_model is not None:
            bst.save_model(save_model)            
     
        """ 
        fscore = self.feature_importance()
        if print_fscore:
            print("Feature Importance:")
            for i in fscore:
                print(i) 
        """
    def predict(self,Xt,load_model=None):
        if load_model and self.bst is None:
            # load model
            self.bst = lgb.Booster(model_file=load_model)
        return self.bst.predict(Xt, num_iteration=self.bst.best_iteration)

