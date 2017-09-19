import os
import csv
from collections import defaultdict
from math import cos
from geohash import decode

cdic={}
ddic={}
def cal_distance(s,t):
    #s,t = sorted([s,t])
    #if (s,t) in cdic:
    #    return cdic[(s,t)]
    if s in cdic:
        lat1,lon1 = cdic[s]
    else:
        lat1,lon1 = decode(s)
        cdic[s] = (lat1,lon1)

    if t in cdic:
        lat2,lon2 = cdic[t]
    else:
        lat2,lon2 = decode(t)
        cdic[t] = (lat2,lon2)

    #lat2,lon2 = decode(t)
    dx = abs(lon1 - lon2)  
    dy = abs(lat1 - lat2)  
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    cdic[(s,t)] = L
    return L
import string
def get_diff(s,t):
    abc = [str(i) for i in range(10)]+list(string.ascii_lowercase)
    dic = {i:c for c,i in enumerate(abc)}
    s,t = sorted([s,t])
    c = 0
    diff = 0
    for i,j in zip(s,t):
        if i!=j:
            diff += len(abc)**(len(s)-c-1)*abs(dic[i]-dic[j])
        c+=1
    return diff

def spatial_distance(inx, base, out):
    if os.path.exists(out) or os.path.exists(inx)==0:
        return

    feas = 'distance,diff'
    header = 'orderid,candidate_loc,label,%s\n'%feas
    fo = open(out,'w')
    fo.write(header)

    dic = {}
    for c,row in enumerate(csv.DictReader(open(base))):
        dic[row['orderid']] = row['geohashed_start_loc']

    for c,row in enumerate(csv.DictReader(open(inx))):
        line = [row['orderid'],row['candidate_loc'],row['label']]
        line.append(str(cal_distance(dic[row['orderid']],row['candidate_loc'])))
        line.append(str(get_diff(dic[row['orderid']],row['candidate_loc'])))
        line = ','.join(line)+'\n'
        fo.write(line)

        if c>0 and c%10000000 == 0:
            print("%s %d rows processed"%(inx,c))
    fo.close()


if __name__ == "__main__":
    spatial_distance(base='comps/mobike/sol_carl/data/tr_sort.csv', 
        inx='comps/mobike/sol_carl/data/tr_norm_count.csv', out='comps/mobike/sol_carl/data/tr_distance.csv')

    spatial_distance(base='comps/mobike/sol_carl/data/va_sort.csv', 
        inx='comps/mobike/sol_carl/data/va_norm_count.csv', out='comps/mobike/sol_carl/data/va_distance.csv')

    spatial_distance(base='../input/train_sort.csv',
        inx='comps/mobike/sol_carl/data/train_norm_count.csv', out='comps/mobike/sol_carl/data/train_distance.csv')

    spatial_distance(base='../input/test_sort.csv',
        inx='comps/mobike/sol_carl/data/test_norm_count.csv', out='comps/mobike/sol_carl/data/test_distance.csv')

