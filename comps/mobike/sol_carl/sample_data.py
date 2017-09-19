import csv
from collections import defaultdict
from utils.pypy_utils.utils import ave,geo_distance,sort_value
from utils.pypy_utils.geohash import float_coord,str_coord
import pickle
import os

def sample_coord_data(data,start_day,end_day,out,min_distance=4000,
    topk=5,is_train=True,counter = {},scounter = defaultdict(int),dist_dic={}):

    if os.path.exists(out):
        return None,None,None

    bsdic = get_consecutive_start_dic(['../input/train_sort.csv','../input/test_sort.csv'],
        'comps/mobike/sol_carl/data/bikes.p','bikeid')
    usdic = get_consecutive_start_dic(['../input/train_sort.csv','../input/test_sort.csv'],
        'comps/mobike/sol_carl/data/users.p','userid')


    h2c = pickle.load(open("comps/mobike/sol_carl/data/h2c.p","rb"))
    c2h = pickle.load(open("comps/mobike/sol_carl/data/c2h.p","rb"))
    candifeas = ["userid","bikeid",'geohashed_start_loc']
    feas = candifeas + ["hour","dow","coord_start"]#,'next_bs','next_us']
    dist_feas = ['ave_dist_ratio','max_dist_ratio','over_ave_dist','over_max_dist']
    all_dist_feas = ["%s-%s"%(i,j) for i in feas for j in dist_feas]

    lenh,lenc,rate = [],[],[]
    fo = open(out,'w')
    head = ",".join(feas + all_dist_feas)
    fo.write("orderid,candidate_loc,label,%s,next_bsx,next_usx,tbsx,tusx\n"%head)
    with open(data) as f:
        for c,row in enumerate(csv.DictReader(f)):
            ymd, hms = row['starttime'].split()
            year,month,day = ymd.split('-')
            day = int(day)
            row["coord_start"] = h2c[row['geohashed_start_loc']]
            if is_train:
                row["coord_end"] = h2c[row['geohashed_end_loc']]
            if 'geohashed_end_loc' not in row:
                row['geohashed_end_loc'] = 'xx'
            row['hour'] = hms.split('-')[0]
            row['dow'] = str(((int(month)-5)*31 + int(day))%7)
            row['next_bs'] = h2c.get(bsdic.get(row['orderid'],('',100000))[0],'')
            row['next_us'] = h2c.get(usdic.get(row['orderid'],('',100000))[0],'')


            if day>end_day:
                break            
            elif is_train==False and day<start_day:
                continue
            elif day>=start_day:
                candidates = set()
                coords = set()
                for fea in candifeas:
                    val = "{}_{}".format(fea,row[fea])
                    dic = counter.get(val,{})
                    keys = sort_value(dic)
                    for d,coord in enumerate(keys):
                        if fea != "userid" and d>topk:
                            continue
                        if geo_distance(float_coord(coord),float_coord(row["coord_start"]))>min_distance:
                            continue
                        coords.add(coord)
                        points = c2h.get(coord,set())
                        candidates = candidates.union(points)      
                lat,lon = float_coord(h2c[row['geohashed_start_loc']])
                for i in [0,0.01,-0.01]:
                    for j in [0,0.01,-0.01]:
                        if abs(i)==0.01 and abs(j)==0.01:
                            continue
                        coord = (lat+i,lon+j)
                        if coord not in c2h:
                            continue
                        if geo_distance(float_coord(coord),float_coord(row["coord_start"]))>min_distance:
                            continue
                        coords.add(coord)
                        ps = c2h.get(coord,[])
                        for p in ps:
                            candidates.add(p)
                tmp = {}
                for fea in feas:
                    val = "%s_%s"%(fea,row[fea])
                    dis = dist_dic.get(val,[0])
                    ad = ave(dis)
                    md = max(dis)
                    tmp[val] = [ad,md]

                if row['next_bs']!='':
                    coords.add(row['next_bs'])
                if row['next_us']!='':
                    coords.add(row['next_us'])


                for i in coords:
                    line=["%s,%s_%s,%d"%(row['orderid'],i[0],i[1],int(row['geohashed_end_loc'] in c2h.get(i,[])))]
                    for fea in feas:
                        val = "%s_%s"%(fea,row[fea])
                        if val in counter:
                            line.append("%.2f"%(counter[val].get(i,0)*1.0/scounter[val]))
                        else:
                            line.append('0')

                    for fea in feas:
                        val = "%s_%s"%(fea,row[fea])
                        ad,md = tmp[val]
                        if md==0 or ad==0:
                            line.extend(['']*4)
                        else:
                            dx = geo_distance(float_coord(i),float_coord(row["coord_start"]))
                            line.extend(["%.2f"%(dx/ad),"%.2f"%(dx/md),str(int(dx>ad)),str(int(dx>md))])
                    if len(line) != 1 + len(feas) + len(all_dist_feas):
                        print(len(line),1 + len(feas) + len(all_dist_feas))
                        print(line)
                        assert 0
                    line.append("%d"%(int(row['next_bs']==i)))
                    line.append("%d"%(int(row['next_us']==i)))
                    if row['next_bs']==i:
                        line.append(str(bsdic.get(row['orderid'],('',10000000))[1]))
                    else:
                        line.append("100000000")
                    if row['next_us']==i:
                        line.append(str(usdic.get(row['orderid'],('',10000000))[1]))
                    else:
                        line.append("100000000")
                    line = ",".join(line)+'\n'
                    fo.write(line)  
                del tmp
                rate.append(int(row['geohashed_end_loc'] in candidates))
                lenh.append(len(candidates))
                lenc.append(len(coords))        
            if c>0 and c%100000 == 0:
                print("samples %d day %d target rate %.2f ave coords %.2f ave hashes %.2f"%(c,day,ave(rate),ave(lenc),ave(lenh)))
            if is_train:
                start = float_coord(row["coord_start"])
                end = float_coord(row["coord_end"])
                dis = geo_distance(start,end)
                for fea in feas:
                    val = "{}_{}".format(fea,row[fea])
                    if val not in counter:
                        counter[val] = defaultdict(int)    
                    counter[val][row["coord_end"]] += 1
                    scounter[val] += 1

                    if val not in dist_dic:
                        dist_dic[val] = []
                    dist_dic[val].append(dis)
                    
            

    fo.close()
    return counter,scounter,dist_dic
    #samples 2400000 day 19 target rate 0.96 ave coords 8.14 ave hashes 341.74
            
def sample_hash_data(inx,out,coord_pred,counter={},scounter=defaultdict(int),startday=16,bads=set(),
    xc=None,xsc=None,threshold=20,max_loc = 30,isva=0,min_count=0):
    """
    normalized count of {xxx, geohashed_end_loc} in history
    """

    if os.path.exists(out):
        return counter,scounter,xc,xsc
    assert 'sort' in inx
    if isva:
        assert "va" in inx or "test" in inx

    bsdic = get_consecutive_start_dic(['../input/train_sort.csv','../input/test_sort.csv'],
        'comps/mobike/sol_carl/data/bikes.p','bikeid')
    usdic = get_consecutive_start_dic(['../input/train_sort.csv','../input/test_sort.csv'],
        'comps/mobike/sol_carl/data/users.p','userid')

    if xc is None or xsc is None:
        xc,xsc =  get_normalize_sloc_counter(['../input/train.csv','../input/test.csv'])
    print("loaded sloc")

    feas = 'gs6_user,gs5_user,coord_start_user,us,ub,userid,bikeid,biketype,dow,geohashed_start_loc,gs6,gs5,gs4,next_bs,next_us,coord_ss'
    candidate_feas = ['coord_start_user','gs5_user','userid','geohashed_start_loc','bikeid']
    header = 'orderid,candidate_loc,label,%s'%feas
    feas = feas.split(',')

    sfeas = 'ub,userid,bikeid,biketype,dow'.split(',')
    header = "%s,%s,coord_score,next_bsx,next_usx,tbsx,tusx\n"%(header,",".join(["s_%s"%i for i in sfeas]))

    fo = open(out,'w')
    fo.write(header)

    rate = []
    coord_score = read_coord_prob(coord_pred)
    coord_candi = read_coord_candi(coord_pred)
    h2c = pickle.load(open("comps/mobike/sol_carl/data/h2c.p","rb"))

    for c,row in enumerate(csv.DictReader(open(inx))):

        ymd, hms = row['starttime'].split()
        #row['hour'] = hms.split('-')[0]
        year,month,day = ymd.split('-')
        row['dow'] = str(((int(month)-5)*31 + int(day))%7)
        row['gs6'] = row['geohashed_start_loc'][:6]
        row['gs5'] = row['geohashed_start_loc'][:5]
        row['gs4'] = row['geohashed_start_loc'][:4]
        #row['ss'] = "%s_%s"%(row['geohashed_start_loc'],bsdic.get(row['orderid'],''))
        row['coord_ss'] = h2c.get(row['geohashed_start_loc'],(0,0)),h2c.get(bsdic.get(row['orderid'],''),(0,0))
        row['next_bs'] = bsdic.get(row['orderid'],('',100000))[0]
        row['next_us'] = usdic.get(row['orderid'],('',100000))[0]
        #row['bs'] = "%s_%s"%(row['geohashed_start_loc'],row['bikeid'])
        row['ub'] = "%s_%s"%(row['userid'],row['bikeid'])
        row['us'] = "%s_%s"%(row['geohashed_start_loc'],row['userid'])
        row["coord_start_user"] = "_".join(list(h2c[row['geohashed_start_loc']])+[row['userid']])
        row['gs6_user'] = row['gs6']+'_'+row['userid']
        row['gs5_user'] = row['gs5']+'_'+row['userid']
        #row["coord_start_bike"] = "_".join(list(h2c[row['geohashed_start_loc']])+[row['bikeid']])
        if 'geohashed_end_loc' not in row:
            row['geohashed_end_loc'] = '-1'

        #if int(day)>13 and isva==0:
        #    break
        if int(day) in bads:
            continue
        if int(day)>startday or isva:
            candidates = set()
            coord_candis = coord_candi.get(row['orderid'],None)
            for fea in candidate_feas:
                val = "%s_%s"%(fea,row[fea])
                dicx = counter.get(val,{})
                keyx = sort_value(dicx)
                d = 0
                for j in keyx:
                    if fea[0]!='u' and d>threshold:
                        break
                    if dicx[j]<min_count:
                        break
                    if coord_candis and h2c[j] not in coord_candis:
                        continue
                    if j!=row['geohashed_start_loc']:
                        candidates.add(j)
                    if isva==0 and len(candidates)>max_loc:
                        break
                    d+=1


            for fea in candidate_feas:
                val = "%s_%s"%(fea,row[fea])
                dicx = xc.get(val,{})
                keyx = sort_value(dicx)
                d = 0
                for j in keyx:
                    if fea[0]!='u' and d>threshold:
                        break
                    if dicx[j]<min_count:
                        break
                    if coord_candis and h2c[j] not in coord_candis:
                        continue
                    if j!=row['geohashed_start_loc']:
                        candidates.add(j)
                    if isva==0 and len(candidates)>max_loc:
                        break
                    d+=1


            if row['next_bs']!='':
                candidates.add(row['next_bs'])
            if row['next_us']!='':
                candidates.add(row['next_us'])
 
            rate.append(int(row['geohashed_end_loc'] in candidates))
            #if row['geohashed_end_loc'] != '-1':
            #    candidates.add(row['geohashed_end_loc'])
            #if len(candidates)>threshold or isva:
            if row['geohashed_end_loc'] in candidates or isva:
                for candidate in candidates:
                    line = [row['orderid'],candidate,str(int(candidate==row['geohashed_end_loc']))]
                    for fea in feas:
                        val = "%s_%s"%(fea,row[fea])
                        if val in counter:
                            line.append(str(counter[val].get(candidate,0)*1.0/scounter[val]))
                        else:
                            line.append('0')
                    for fea in sfeas:
                        val = "%s_%s"%(fea,row[fea])
                        if val in xc:
                            line.append(str(xc[val].get(candidate,0)*1.0/xsc[val]))
                        else:
                            line.append('0')
                    line.append("%.4f"%(coord_score.get(row['orderid'],{}).get(h2c[candidate],-1000)))
                    line.append("%d"%(int(row['next_bs']==candidate)))
                    line.append("%d"%(int(row['next_us']==candidate)))
                    if row['next_bs']==candidate:
                        line.append(str(bsdic.get(row['orderid'],('',10000000))[1]))
                    else:
                        line.append("100000000")
                    if row['next_us']==candidate:
                        line.append(str(usdic.get(row['orderid'],('',10000000))[1]))
                    else:
                        line.append("100000000")
                    #line.append(row['dow'])
                    line = ','.join(line)+'\n'
                    fo.write(line)
        if isva==0:#row['geohashed_end_loc'] != '-1':
            for fea in feas:
                val = "%s_%s"%(fea,row[fea])
                if val not in counter:
                    counter[val] = defaultdict(int)
                counter[val][row['geohashed_end_loc']] += 1
                scounter[val] += 1

        if c>0 and c%100000 == 0:
            print("%s %d day %s rows processed, cover rate %.3f"%(inx,c,day,ave(rate)))
    fo.close()
    print("sample hash data done")
    return counter,scounter,xc,xsc

def read_coord_candi(coord_pred):
    f = open(coord_pred)
    coord_score = {}
    for c,row in enumerate(f):
        xx = row.strip().split(',')
        if xx[0] not in coord_score:
            coord_score[xx[0]]=set()
        for i in xx[1:]:
            coord_score[xx[0]].add(tuple(i.split('_')))
    f.close()
    return coord_score

def read_coord_prob(coord_pred):
    f = open(coord_pred)
    coord_score = {}
    for c,row in enumerate(csv.DictReader(f)):
        if row['orderid'] not in coord_score:
            coord_score[row['orderid']]={}
        coord_score[row['orderid']][tuple(row['candidate_loc'].split('_'))] = float(row['prob'])
    f.close()
    return coord_score

def get_consecutive_start_dic(ins,op,fea):
    def _time_diff(t1,t2):
        t1 = t1.split(':')
        t1 = [float(i) for i in t1]
        t2 = t2.split(':')
        t2 = [float(i) for i in t2]
        h1,m1,s1 = t1
        h2,m2,s2 = t2
        return ((h2-h1)*60+(m2-m1))*60+s2-s1
    #op = "comps/mobike/sol_carl/data/ssdic.p"
    if os.path.exists(op):
        return pickle.load(open(op,'rb'))

    dic={}
    for inx in ins:
        for c,row in enumerate(csv.DictReader(open(inx))):
            if row[fea] not in dic:
                dic[row[fea]] = []
            ymd, hms = row['starttime'].split()
            #row['hour'] = hms.split('-')[0]
            dic[row[fea]].append((row['orderid'],row['geohashed_start_loc'],ymd,hms))
            if c>0 and c%100000 == 0:
                print("%s %s %d rows processed"%(op,inx,c))

    bdic = {}
    for bike,ll in dic.items():
        for c,l in enumerate(ll[:-1]):
            k = l[0]
            if ll[c+1][2] == ll[c][2]:
                bdic[k] = (ll[c+1][1],_time_diff(ll[c][3],ll[c+1][3]))

    pickle.dump(bdic,open(op,'wb'))
    return bdic

def get_normalize_sloc_counter(inxs,counter={},scounter=defaultdict(int)):
    """
    normalized count of {xxx, geohashed_end_loc} in history
    """
    feas = 'ub,userid,bikeid,biketype,hour,dow'
    op = "comps/mobike/sol_carl/data/all_sc.p"
    if os.path.exists(op):
        return pickle.load(open(op,'rb'))
    candidate_feas = ['bikeid','userid']
    feas = feas.split(',')
    rate = 0
    for inx in inxs:
        for c,row in enumerate(csv.DictReader(open(inx))):

            ymd, hms = row['starttime'].split()
            row['hour'] = hms.split('-')[0]
            year,month,day = ymd.split('-')
            row['dow'] = str(((int(month)-5)*31 + int(day))%7)
            row['ub'] = "%s_%s"%(row['userid'],row['bikeid'])

            for fea in feas:
                val = "%s_%s"%(fea,row[fea])
                if val not in counter:
                    counter[val] = defaultdict(int)
                counter[val][row['geohashed_start_loc']] += 1
                scounter[val] += 1
            for fea in candidate_feas:
                val = "%s_%s"%(fea,row[fea])
                if 'geohashed_end_loc' in row and row['geohashed_end_loc'] in counter[val]:
                    rate+=1
                    break
            if c>0 and c%100000 == 0:
                print("%s %d day %s rows processed, cover rate %.3f"%(inx,c,day,(rate*1.0/c)))
    pickle.dump((counter,scounter),open(op,'wb'))
    return counter,scounter

def rm_low_freq(inx,out,base,bar=10):
    if os.path.exists(out):
        return 
    bdic = read_base(base)
    fo = open(out,'w')
    f = open(inx)
    fo.write(f.readline())
    last = ''
    lines = []
    labels = 0
    for c,row in enumerate(csv.DictReader(open(inx))):
        if last!='' and last!=row['orderid']:
            if labels==1:
                for line in lines:
                    fo.write(line)
            lines,labels = [],0
        line = f.readline()
        if bdic.get(row['candidate_loc'],0)>bar:
            lines.append(line)
            labels += int(row['label'])
        last = row['orderid']
    if last!='' and last!=row['orderid']:
        if labels==1:
            for line in lines:
                fo.write(line)
    f.close()
    fo.close()

def read_base(base):
    out = base.split('/')[-1].replace(".csv",".p")
    out = "comps/mobike/sol_carl/data/%s"%out
    if os.path.exists(out):
        return pickle.load(open(out))
    bdic = defaultdict(int)
    for c,row in enumerate(csv.DictReader(open(base))):
        bdic[row['geohashed_end_loc']]+=1
    pickle.dump(bdic,open(out,'w'))
    return bdic
 
