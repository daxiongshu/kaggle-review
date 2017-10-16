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
        print(self.params)
        early_stopping_rounds = self.params.get('early_stopping_rounds',None)
        params = self.params.copy()

        dtrain = lgb.Dataset(X, y)

        if Xt is not None:
            dvalid = lgb.Dataset(Xt, yt, reference=dtrain)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = lgb.train(params, dtrain,
                num_boost_round=num_round,
                valid_sets=dvalid,
                early_stopping_rounds=early_stopping_rounds)
        else:
            bst = lgb.train(params,dtrain,
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

    def bag_fit_predict(self,X,y,Xt,obj=None,feval=None,folds=4,
        stratified=True,ydim=1,shuffle=True):
        #X,y,Xt = np.array(X),np.array(y),np.array(Xt)
        print(self.params)
        num_round = self.params.get('num_round',1000)
        early_stopping_rounds = self.params['early_stopping_rounds']
        maximize = self.params.get('maximize',False)
        def _get_split(X,y,stratified):
            import sklearn
            if stratified:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=126)
                for tr,te in kf.split(X,y):
                    yield tr,te
            else:
                kf = sklearn.model_selection.KFold(n_splits=folds, shuffle=shuffle, random_state=126)
                for tr,te in kf.split(X):
                    yield tr,te
        if ydim==1:
            yp = np.zeros(Xt.shape[0])
        else:
            yp = np.zeros([Xt.shape[0],ydim])
        scores = []
        for k,(tr,te) in enumerate(_get_split(X,y,stratified)):
            Xtr,Xte = X[tr],X[te]
            ytr,yte = y[tr],y[te]
            score = self.fit(Xtr,ytr,Xt=Xte,yt=yte,obj=obj,feval=feval)
            scores.append(score)
            yp += self.predict(Xt)
            del self.bst
            print("fold {} done\n\n".format(k))
        return yp/folds
