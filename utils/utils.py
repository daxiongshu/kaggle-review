import time
import numpy as np
import os
import random
import sys

_start = time.time()
try:
    import psutil
    _summ = psutil.virtual_memory()
except:
    pass

def print_mem_time(tag):
    print(tag," time %.2f seconds"%(time.time()-_start), end='')
    try:
        summ = psutil.virtual_memory()
        print(" Used: %.2f GB %.2f%%"%((summ.used)/(1024.0*1024*1024),summ.percent))
    except:
        print()

def split(flags):
    if os.path.exists(flags.split_path):
        return np.load(flags.split_path).item()
    folds = flags.folds
    path = flags.input_path
    random.seed(6)
    img_list = ["%s/%s"%(path,img) for img in os.listdir(path)]
    random.shuffle(img_list)
    dic = {}
    n = len(img_list)
    num = (n+folds-1)//folds
    for i in range(folds):
        s,e = i*num,min(i*num+num,n)
        dic[i] = img_list[s:e]
    np.save(flags.split_path,dic)
    return dic
