import numpy as np
import pandas as pd
import os
import re
def clean(flags):
    print("clean stage1 test file ...")
    path = flags.input_path
    name = "%s/stage1_solution_filtered.csv"%path
    base = pd.read_csv(name)
    base['Class'] = np.argmax(base[['class%d'%i for i in range(1,10)]].values,axis=1)+1
    for name in ["test_variants"]:
        name = "%s/%s"%(path,name)
        if os.path.exists(name+'_filter'):
            continue
        s = pd.read_csv(name)
        s = pd.merge(base[['ID','Class']],s,on='ID',how='left')
        name = name+'_filter'
        print(name,s.shape)
        s.to_csv(name,index=False)

    name = "%s/test_text"%path
    if os.path.exists(name+'_filter'):
        return

    rows = set(list(base['ID'].to_dict().values()))
    oname = name + '_filter'
    f = open(name,encoding="utf-8")
    fo = open(oname,'w',encoding="utf-8")
    fo.write(f.readline())
    for line in f:
        xx = line.split('||')[0]
        if int(xx) in rows:
            fo.write(line)
    f.close()
    fo.close()
