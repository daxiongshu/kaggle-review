from utils.pypy_utils.geohash import decode,str_coord
from utils.pypy_utils.utils import geo_distance,read_fscore,sort_value
import os
import pickle
import csv
#from utils.draw.sns_draw import distribution

def build_hash_to_coord(paths):
    if os.path.exists("comps/mobike/sol_carl/data/h2c.p") and os.path.exists("comps/mobike/sol_carl/data/c2h.p"):
        return
    h2c,c2h = {},{}
    for path in paths:
        for c,row in enumerate(csv.DictReader(open(path))):
            for tag in ["geohashed_end_loc","geohashed_start_loc"]:
                if tag not in row:
                    continue
                h = row[tag]
                if h not in h2c:
                    coord = str_coord(decode(h))
                    h2c[h] = coord
                    #lat,lon = int(lat+0.5),int(lon+0.5)
                    if coord not in c2h:
                        c2h[coord] = set()
                    c2h[coord].add(h)
            if c>0 and c%100000 == 0:
                print(path,c)
    print(len(h2c),len(c2h))
    pickle.dump(h2c,open("comps/mobike/sol_carl/data/h2c.p","wb"))
    pickle.dump(c2h,open("comps/mobike/sol_carl/data/c2h.p","wb"))                

def find_neighbor():
    h2c = pickle.load(open("comps/mobike/sol_carl/data/h2c.p","rb"))
    c2h = pickle.load(open("comps/mobike/sol_carl/data/c2h.p","rb"))
    print(len(h2c),len(c2h))
    lc = [len(c2h[i])  for i in c2h]
    #distribution(lc)
    #point = list(h2c.keys())[0]
    point = "wx4snhx"
    print("hash", point, h2c[point])
    lat,lon = h2c[point]
    #lat,lon = int(lat+0.5),int(lon+0.5) 
    points = c2h[(lat,lon)]
    for la in [lat-0.01,lat,lat+0.01]:
        for lo in [lon-0.01,lon,lon+0.01]:
            coord = (la,lo)
            points = c2h.get(coord,[])
            for p in points:
                d = geo_distance(h2c[p],(lat,lon))
                print(coord,p,d)

def coord2hash(hash_data,coord_data,out,coord_fscore,bar=0,topk=100):
    if os.path.exists(out):
        return
    fscore = read_fscore(coord_fscore)
    fscore = {i:fscore[i] for i in fscore if fscore[i]>bar and '-' in i}
    feas = sort_value(fscore)[:topk] 
    print(feas)
    print(len(feas)) 
    h2c = pickle.load(open("comps/mobike/sol_carl/data/h2c.p","rb"))
    c2h = pickle.load(open("comps/mobike/sol_carl/data/c2h.p","rb"))

    fo = open(out,'w') 
    fo.write("orderid,label,candidate_loc,%s\n"%(','.join(feas)))
    dic = {}
    for row in csv.DictReader(open(coord_data)):
        if row['orderid'] not in dic:
            dic[row['orderid']] = {}
        dic[row['orderid']][row['candidate_loc']]={}
        for fea in feas:
            dic[row['orderid']][row['candidate_loc']][fea] = row[fea]
    print("read %s done"%coord_data)

    for c,row in enumerate(csv.DictReader(open(hash_data))):
        line = [row['orderid'],row['label'],row['candidate_loc']]
        coord = "_".join(h2c.get(row['candidate_loc'],('','')))
        if row['orderid'] not in dic or coord not in dic[row['orderid']]:
            line.extend(['']*len(feas))
        else:
            for fea in feas:
                line.append(dic[row['orderid']][coord][fea])
        line = ",".join(line)+'\n'
        fo.write(line)
        if c>0 and c%100000 == 0:
            print(c,"write",out)
    fo.close()

    
