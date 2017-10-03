import pandas as pd
import sys
import numpy as np

def replace(s,n):
    seen = pd.read_csv(s)
    unseen = pd.read_csv(n)
    te = pd.read_csv('../input/stage2_test_variants.csv')
    tr = pd.read_csv('../input/training_variants')
    unseen = pd.merge(unseen,te,on='ID',how='right')
    seen = pd.merge(seen,te,on='ID',how='right')
    mask = seen.Gene.isin(tr.Gene)
    cols = ['class%d'%i for i in range(1,10)]
    seen.loc[~mask,cols] = 0

    mask = unseen.Gene.isin(tr.Gene)
    unseen.loc[mask,cols] = 0

    assert (unseen['ID']==seen['ID']).all()
    seen[cols] = seen[cols] + unseen[cols]

    seen[cols+['ID']].to_csv('mix.csv',index=False)


if __name__ == "__main__":
    s,n = sys.argv[1:]
    replace(s,n)
