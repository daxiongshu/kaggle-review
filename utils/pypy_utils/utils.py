from math import cos,log
import pickle
import os

def load_pickle(data,name,default):
    if data is not None:
        return data
        
    if os.path.exists(name):
        return pickle.load(open(name,'rb'))
    else:
        return default

def save_pickle(data,name):
        
    if os.path.exists(name)==0:
        pickle.dump(data,open(name,'wb'))

def read_fscore(name):
    dic = {}
    with open(name) as f:
        for line in f:
            xx = line.strip().split(',')
            fea = xx[0].strip()[1:]
            score = xx[1].strip()[:-1]       
            dic[fea[1:-1]] = float(score)
    return dic     

def logloss(y,yp):
    return -(y*log(yp)+(1-y)*log(1-yp))

def ave(x):
    if len(x)==0:
        return 0
    return sum(x)*1.0/len(x)

def sort_value(dic):
    return sorted(dic, key=dic.get, reverse=True)

def geo_distance(coord1,coord2):
    lat1,lon1 = coord1
    lat2,lon2 = coord2
    dx = abs(lon1 - lon2)
    dy = abs(lat1 - lat2)
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

def apk(actual, predicted, k=3):
    #print(actual, predicted)
    #assert 0
    if len(predicted)>k:
        predicted = predicted[:k]
    if not actual:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(actual), k)

