import sklearn
import pandas as pd
import numpy as np

class FFMEncoder(sklearn.base.BaseEstimator):
    def fit(self, X, y=None,N=None):
        """
        x is a dataframe of numpy array
        x should not have target column or id column
        """
        assert isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)
        dic = {}
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=['col%d'%i for i in range(X.shape[1])])            
        for col in X.columns.values:
            #if X[col].dtype!=np.float32:
            xx = X[col].unique()
            #xx = sorted(X[col].unique())
            m = 1 if N is None else len(xx)//N
            if m==0:
                m = 1
            #dic[col] = {i:(c if N is None else c%N) for c,i in enumerate(xx)}
            dic[col] = {i:c//m for c,i in enumerate(xx)}
            print(col,len(xx),max(dic[col].values()))
        self.dic = dic
        self.N = N
        return self

    def transform(self, X):
        assert isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)
        #assert X.shape[1] == len(self.dic)
        dic = self.dic
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=['col%d'%i for i in range(X.shape[1])])
        sx = 1 
        cols = [i for i in dic]
        for col in cols:
            X[col] = X[col].apply(lambda x: int(dic[col][x]+sx) if x in dic[col] else 0)
            sx += int(max(dic[col].values())+1)
        return X[cols]
