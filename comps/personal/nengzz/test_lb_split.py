import sys
import pandas as pd

def test1(name):
    s1 = pd.read_csv('../input/test_variants')
    s2 = pd.read_csv('../input/stage2_test_variants.csv')
    s1 = pd.merge(s1,s2,on= ["Gene", "Variation"],how='inner')
    s = pd.read_csv(name)
    mask = s['ID'].isin(s1['ID_y'])
    cols = ['class%d'%i for i in range(1,10)]
    print(s.shape,s[~mask].shape)
    s.loc[~mask,cols] = 0.1
    s.to_csv('sub.csv',index=False)

def test2():
    s1 = pd.read_csv('../input/test_variants')
    s3 = pd.read_csv('../input/test_variants_filter')
    s1 = pd.merge(s1,s3[['ID','Class']],on='ID',how='left').fillna(1)

    s2 = pd.read_csv('../input/stage2_test_variants.csv')
    s1 = pd.merge(s1,s2,on= ["Gene", "Variation"],how='inner')
    s1['ID'] = s1['ID_y']
    s2 = pd.merge(s1[['ID','Class']],s2,on='ID',how='right').fillna(1)
    yp = onehot_encode(s2['Class'].values-1)

    for i in range(1,10):
        s2['class%d'%i] = yp[:,i-1]
    cols = ['class%d'%i for i in range(1,10)]
    mask = s2['ID'].isin(s1['ID_y'])
    s2.loc[~mask,cols] = 0.1

    s2['ID'] = s2['ID'].astype(int)
    cols = ['ID']+['class%d'%i for i in range(1,10)]
    s2[cols].to_csv('sub.csv',index=False)

def test3(name):
    sub = pd.read_csv(name)
    s1 = pd.read_csv('../input/test_variants')
    s3 = pd.read_csv('../input/test_variants_filter')
    s1 = pd.merge(s1,s3[['ID','Class']],on='ID',how='left').fillna(1)

    s2 = pd.read_csv('../input/stage2_test_variants.csv')
    s1 = pd.merge(s1,s2,on= ["Gene", "Variation"],how='inner')
    s1['ID'] = s1['ID_y']
    s2 = pd.merge(s1[['ID','Class']],s2,on='ID',how='right').fillna(1)
    yp = onehot_encode(s2['Class'].values-1)

    for i in range(1,10):
        s2['class%d'%i] = yp[:,i-1]
    cols = ['class%d'%i for i in range(1,10)]

    mask = s2['ID'].isin(s1['ID_y'])
    s2.loc[~mask,cols] = 0

    s3 = pd.merge(s2[['ID']],sub,on='ID',how='left')
    assert (s2['ID']==s3['ID']).all()
    s3.loc[mask,cols] = 0
    #s2.set_index('ID',inplace=True)

    s2[cols] = s3[cols]+s2[cols]

    s2['ID'] = s2['ID'].astype(int)
    cols = ['ID']+['class%d'%i for i in range(1,10)]
    s2[cols].to_csv('sub.csv',index=False)

import numpy as np

def onehot_encode(y,n=None):
    """
    Input:
        y 1d array of lenth B, elements from {0,1,..,n-1}
        n: the number of classes
    Return: yp 2d aray [B,N]
    """
    if n is None:
        n = np.max(y)+1
    yp = np.zeros([y.shape[0],n])
    x = np.arange(y.shape[0])
    yp[x,y.astype(int)] = 1
    return yp
if __name__ == "__main__":
    name = sys.argv[1]
    #test1(name)
    #test2()
    test3(name)

