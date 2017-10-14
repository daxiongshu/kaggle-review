"""
xgboost for classification & regression
"""
import xgboost as xgb
import numpy as np
from utils.pypy_utils.utils import sort_value
 
class xgb_model(object):
    
    def __init__(self,params):
        self.params = params
        self.bst = None

    def fit(self,X,y,Xt=None,yt=None,
        load_model=None,save_model=None,
        obj=None,feval=None,print_fscore=True,evalx=None):
        print(X.shape,y.shape)

        num_round = self.params.get('num_round',100)
        early_stopping_rounds = self.params.get('early_stopping_rounds',None)
        maximize = self.params.get('maximize',False)
        dtrain = xgb.DMatrix(X, y)
        vb = self.params.get('verbose_eval',1)
        if Xt is not None:
            dvalid = xgb.DMatrix(Xt, yt)
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                early_stopping_rounds=early_stopping_rounds,verbose_eval=vb,
                xgb_model=load_model,obj=obj,feval=feval,maximize=maximize)
        else:
            watchlist = [(dtrain, 'train')]
            bst = xgb.train(self.params, dtrain, num_round, evals = watchlist,
                verbose_eval=vb,xgb_model=load_model,obj=obj,feval=feval)
        self.bst = bst
        if save_model is not None:
            bst.save_model(save_model)            
       
        fscore = self.feature_importance()
        if print_fscore:
            print("Feature Importance:")
            for i in fscore:
                print(i) 
        if Xt is not None and evalx is not None:
            yp = self.predict(Xt)
            score = evalx(yt,yp)
            print(score)
            return score
        return 0

    def bag_fit_predict(self,X,y,Xt,obj=None,feval=None,folds=4,stratified=True,ydim=1,evalx=None):
        assert (self.params['colsample_bytree'] < 1 or self.params['subsample'] < 1)
        #X,y,Xt = np.array(X),np.array(y),np.array(Xt)
        num_round = self.params.get('num_round',1000)
        early_stopping_rounds = self.params['early_stopping_rounds']
        maximize = self.params.get('maximize',False)

        def _get_split(X,y,stratified):
            import sklearn
            if stratified:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=126)
                for tr,te in kf.split(X,y):
                    yield tr,te
            else:
                kf = sklearn.model_selection.KFold(n_splits=folds, shuffle=True, random_state=126)
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
            score = self.fit(Xtr,ytr,Xt=Xte,yt=yte,obj=obj,feval=feval,
                print_fscore=False,evalx=evalx)
            scores.append(score)
            yp += self.predict(Xt)
            print("fold {} done\n\n".format(k))
        if evalx is not None:
            print(scores,np.mean(scores))
        return yp/folds
        

    def predict(self,Xt,load_model=None):
        dtest = xgb.DMatrix(Xt)
        if load_model and self.bst is None:
            self.bst = xgb.Booster(self.params,model_file=load_model)
        return self.bst.predict(dtest)

    def feature_importance(self):
        fscore = self.bst.get_fscore()
        feas = sort_value(fscore)
        return [(fea,fscore[fea]) for fea in feas]        
