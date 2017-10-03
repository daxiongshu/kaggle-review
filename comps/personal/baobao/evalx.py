import numpy as np
import pandas as pd
import sys
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
def cross_entropy(y,yp):
    yp[yp>0.99999] = 0.99999
    yp[yp<1e-5] = 1e-5
    return np.mean(-np.log(yp[range(yp.shape[0]),y.astype(int)]))

def eval(name,clip=False,bar=0.9):
    base = pd.read_csv('../input/stage1_solution_filtered.csv')
    base['Class'] = np.argmax(base[['class%d'%i for i in range(1,10)]].values,axis=1)
    sub = pd.read_csv(name)
    #sub = pd.merge(sub,base[['ID','Class']],on="ID",how='right')
    #print(sub.head())
    y = base['Class'].values
    yp = sub[['class%d'%i for i in range(1,10)]].values
    if clip:
        yp = np.clip(yp,(1.0-bar)/8,bar)
        yp = yp/np.sum(yp,axis=1).reshape([yp.shape[0],1])
    print(name,cross_entropy(y,yp),multiclass_log_loss(y,yp))
    for i in range(9):
        y1 = y[y==i]
        yp1 = yp[y==i]
        print(i,y1.shape,cross_entropy(y1,yp1),multiclass_log_loss(y1,yp1))

if __name__ == "__main__":
    name = sys.argv[1]
    if len(sys.argv)>2:
        bar = float(sys.argv[2])
        eval(name,True,bar)
    else:
        eval(name)
