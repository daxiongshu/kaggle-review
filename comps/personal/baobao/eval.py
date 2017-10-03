from utils.np_utils.encoder import onehot_encode
from utils.np_utils.utils import cross_entropy
from comps.personal.personal_db import personalDB
import numpy as np
import pandas as pd
from comps.personal.baobao.fe import to4c
def eval(flags):
    name = flags.pred_path
    yp = pd.read_csv(name)
    classes = len([i for i in yp.columns.values if 'class' in i])
    yp = yp[['class%d'%i for i in range(1,classes+1)]].values
    myDB = personalDB(flags,name="full")
    if "stage1" in name:
        y=myDB.data['test_variants_filter']['Class']-1
    else:
        myDB.get_split()
        va = myDB.split[flags.fold][1]
        y = np.argmax(myDB.y[va],axis=1)
    if np.max(y)>classes:
        y = np.argmax(to4c(onehot_encode(y)),axis=1)
    score = cross_entropy(y,yp)
    print(name,score,'\n')
