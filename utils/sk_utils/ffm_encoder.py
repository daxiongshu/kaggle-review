import sklearn
import pandas as pd
import numpy as np

class FFMEncoder(sklearn.base.BaseEstimator):
    def fit(self, X, y=None):
        """
        x is a dataframe of numpy array
        x should not have target column or id column
        """
        assert isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)
        dic = {}
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=['col%d'%i for i in range(X.shape[1])])            
        for col in X.columns.values:
            dic[col] = {i:c for c,i in enumerate(X[col].unique())}
        self.dic = dic
        return self

    def transform(self, X):
        assert isinstance(X,pd.DataFrame) or isinstance(X,np.ndarray)
        assert X.shape[1] == len(self.dic)
        dic = self.dic
        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=['col%d'%i for i in range(X.shape[1])])
        sx = 1 
        for col in X.columns.values:
            X[col] = X[col].apply(lambda x: dic[col][x]+sx if x in dic[col] else 0)
            sx += len(dic[col])
        return X
