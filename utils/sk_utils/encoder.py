from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
def onehot_encode(tr,te,cols=None):
    if cols is None:
        cols = [i for i in tr.columns.values if i in te.columns.values]
    vec = DictVectorizer()
    for col in cols:
        tr[col] = tr[col].map(str)
        te[col] = te[col].map(str)
    print("start fitting")
    X = vec.fit_transform(tr[cols].T.to_dict().values())
    Xt = vec.transform(te[cols].T.to_dict().values())
    print("done fitting",X.shape,Xt.shape)
    return X,Xt

def onehot_encode_bar(tr,te,cols=None,bar=10000):
    if cols is None:
        cols = [i for i in tr.columns.values if i in te.columns.values]
    vec = DictVectorizer()
    cat,num = [],[]
    for col in cols:
        nu = tr[col].unique().shape[0]
        if (nu<bar and nu>2) or tr[col].dtype=='object':
            cat.append(col)
            tr[col] = tr[col].map(str)
            te[col] = te[col].map(str)
        else:
            num.append(col)
    print("start fitting num of cat features:",len(cat))
    X = vec.fit_transform(tr[cat].T.to_dict().values())
    Xt = vec.transform(te[cat].T.to_dict().values())
    print("done fitting",X.shape,Xt.shape)
    X = sparse.hstack([X,tr[num].values],format='csr')
    Xt = sparse.hstack([Xt,te[num].values],format='csr') 
    return X,Xt
