import sys
import csv
import os
def sort_value(dic):
    return sorted(dic, key=dic.get, reverse=True)

def post(inx,k=3):
    out = inx.replace('.csv','_sub.csv')
    #if os.path.exists(out):
    #    return
    fo = open(out,'w')
    last = ''
    pred = {}
    for c,row in enumerate(csv.DictReader(open(inx))):   
        if last != '' and row['orderid'] != last:
            pred = ','.join(sort_value(pred)[:3])
            fo.write('%s,%s\n'%(last,pred))
            pred = {}
        yp = float(row['prob'])
        pred[row['candidate_loc']] = yp
        last = row['orderid']
        if  c % 10000000 == 0 and c>0:
            print c#, 'log loss', score/(c+1), 'm12 apk', apks/dc
    pred = ','.join(sort_value(pred)[:k])
    fo.write('%s,%s\n'%(row['orderid'],pred))
    fo.close()

def post_ffm(inx):
    out = inx.replace('.csv','_sub.csv')
    idx = "comps/mobike/sol_carl/data/va_20-24.id" 
    last = ''
    pred = {}
    f = open(inx)
    fo = open(out,'w')
    for c,row in enumerate(csv.DictReader(open(idx))):
        line = f.readline()
        row['prob'] = line.strip()
        if last != '' and row['orderid'] != last:
            pred = ','.join(sort_value(pred)[:3])
            fo.write('%s,%s\n'%(last,pred))
            pred = {}
        yp = float(row['prob'])
        pred[row['candidate_loc']] = yp
        last = row['orderid']
        if  c % 10000000 == 0 and c>0:
            print c#, 'log loss', score/(c+1), 'm12 apk', apks/dc
    pred = ','.join(sort_value(pred)[:3])
    fo.write('%s,%s\n'%(row['orderid'],pred))
    fo.close()
    f.close()

if __name__ == "__main__":
    name = sys.argv[1]
    if len(sys.argv)>2:
        num = int(sys.argv[2])
    else:
        num = 3
    if "fm" not in name:
        post(sys.argv[1],num)
    else:
        post_ffm(sys.argv[1])
