from utils.pypy_utils.utils import logloss,apk,ave
import csv

def topk(true,pred,k):
    return float(true in pred[:k])

def eval(bdata,data,sub,label,idx,candidate,k=3,out=None):
    dic = {}
    seenuser = set()
    tr_user = set()
    if bdata is not None:
        for row in csv.DictReader(open(bdata)):
            tr_user.add(row['userid'])
        for row in csv.DictReader(open(bdata.replace('tr_','va_'))):
            if row['userid'] in tr_user:
                seenuser.add(row[idx])
    
    days = {}
    for row in csv.DictReader(open("comps/mobike/sol_carl/data/va_sort.csv")):
        ymd, hms = row['starttime'].split()
        #row['hour'] = hms.split('-')[0]
        year,month,day = ymd.split('-')
        row['dow'] = str(((int(month)-5)*31 + int(day))%7)

        days[row['orderid']] = row['dow'] 

    dayscore = {}
    for i in list(days.values()):
        dayscore[i] = []
    for row in csv.DictReader(open(data)):
        if label is None or int(row[label]):
            dic[row[idx]] = row[candidate]
    scores = 0
    seen_scores = 0
    seen_sk = 0
    sk = 0
    f = open(sub)
    sc = 1
    if out:
        fo = open(out,'w')
        fo.write("orderid,apk\n") 
    for c,line in enumerate(f):
        xx = line.strip().split(',')
        pred = xx[1:]
        real = dic.get(xx[0],"")
        s1,s2 = apk([real],pred,k),topk(real,pred,k)
        if xx[0] in days:
            dayscore[days[xx[0]]].append(s1)
        if xx[0] in seenuser:
            seen_scores += s1
            seen_sk += s2
            sc+=1
        scores += s1
        sk += s2
        if out:
            fo.write("%s,%.3f\n"%(xx[0],s1))
        #print(real,pred,scores)
        #break
        if c>0 and c%100000 == 0:
            print(c,"apk %.4f"%(scores/c),"top%d %.4f"%(k,sk/c),'seen user apk %.4f'%(seen_scores/sc),'seen user topk %.4f'%(seen_sk/sc),"seen",sc)
            if xx[0] in days:
                print([(i,"%.2f"%ave(j)) for i,j in dayscore.items()])
    print(c,"apk %.4f"%(scores/c),"top%d %.4f"%(k,sk/c),'seen user apk %.4f'%(seen_scores/sc),'seen user topk %.4f'%(seen_sk/sc),"seen",sc)
    if xx[0] in days:
        print([(i,"%.2f"%ave(j)) for i,j in dayscore.items()])
    f.close()
    if out:
        fo.close()
