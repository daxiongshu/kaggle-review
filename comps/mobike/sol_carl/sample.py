import csv
from random import random
import os

def sample(name,ratio=0.05):
    oname = name.replace('.csv','_sample.csv')
    if os.path.exists(oname):
        return
    num = int(1/ratio)
    fo = open(oname,'w')
    f = open(name)
    fo.write(f.readline())
    dic = {}
    for row in csv.DictReader(open('comps/mobike/sol_carl/data/va_label.csv')):
        dic[row['orderid']] = row['geohashed_end_loc']
    for c,line in enumerate(f):
        xx = line.split(',')
        orderid,loc,label = 0,1,2
        idx = hash(xx[orderid])%100000
        if idx%num==0:#random()<ratio:
            xx[label] = str(int(xx[loc]==dic[xx[orderid]]))
            line = ",".join(xx)
            fo.write(line)
        if c%10000000 == 0:
            print(name,c)
    f.close()
    fo.close()

if __name__ == "__main__":
    path = "comps/mobike/sol_carl/data"
    for i in ['va_norm_count.csv','va_distance.csv']:
        sample("%s/%s"%(path,i))

