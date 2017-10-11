from math import cos,log
import pickle
import os
import csv
from collections import defaultdict

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

def sort_value(dic,descending=True):
    return sorted(dic, key=dic.get, reverse=descending)

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

def csv2ffm(inx,out,id_col,y_col,fea_dic={},update=False,ignorenan=True,mtr=None,bl=0,bar=0,na=''):
    print("csv2ffm",out)
    if os.path.exists(out):
        return fea_dic
    f = open(inx)
    head = f.readline().strip()
    fields = head.split(',')
    f.close()
    fields = [i for i in fields if i not in [id_col,y_col]]
    field_dic = {i:c for c,i in enumerate(fields)}

    fo = open(out,'w')
    cq = 0
    with open(inx) as f:
        for c,row in enumerate(csv.DictReader(f)):
            line = [row.get(y_col,'0')]
            for field in fields:
                if ignorenan and row[field]==na:
                    continue
                #if '.' in row[field] and len(row[field])>5:
                #    row[field] = row[field][:-1]
                val = "%s-%s"%(field,row[field])
                if mtr is not None and (val not in mtr or abs(mtr[val]-bl)<bar):
                    cq += 1
                    continue
                if val not in fea_dic:
                    if update:
                        fea_dic[val] = len(fea_dic)+1
                    else:
                        continue
                m = field_dic[field]
                n = fea_dic.get(val,'0')
                line.append("%s:%s:1"%(m,n))
            line = " ".join(line)
            fo.write(line+'\n')
            if c>0 and c%100000 ==  0:
                print(c,out,'written','ignore',cq)
    fo.close()
    return fea_dic

def mean_target_rate(name,out,idcol,ycol):
    if os.path.exists(out):
        return pickle.load(open(out,'rb'))
    yc,cc = defaultdict(float),defaultdict(float)
    for c,row in enumerate(csv.DictReader(open(name))):
        y = float(row[ycol])
        for i in row:
            if i in [idcol,ycol]:
                continue
            v = "%s-%s"%(i,row[i])
            yc[v] += y
            cc[v] += 1.0

        if c>0 and c%100000 == 0:
            print("rows %d len_cc %d"%(c,len(cc)))
    for i in yc:
        yc[i] = yc[i]/cc[i]
    pickle.dump(yc,open(out,'wb'))
    return yc

def ffm2svm(inx,out):
    if os.path.exists(out):
        return 
    fo = open(out,'w')
    f = open(inx)
    for line in f:
        xx = line.strip().split()
        fea = [int(i.split(':')[1]) for i in xx[1:]]
        fea = sorted(fea)
        line = ["%d:1"%i for i in fea]
        fo.write("%s %s\n"%(xx[0]," ".join(line)))
    f.close()
    fo.close()
