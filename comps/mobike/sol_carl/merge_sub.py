"""
import pandas as pd
def merge_sub(sub,base):
    sb = pd.read_csv(base,header=False)
    sb.columns = "orderid,candidate_loc,prob".split(',')
    s = pd.read_csv(sub,header=False)
    sb.columns = "orderid,candidate_loc,prob".split(',')
    s = pd.merge(s,sb,on="orderid",how='right')
"""
f = open('comps/mobike/sol_carl/sub_sub.csv')
dic = {}
for line in f:
    idx = line.split(',')[0]
    dic[idx] = line
f.close()
fo = open('comps/mobike/sol_carl/result.csv','w')
f = open('../input/sample_submission.csv')
c = 0
for line in f:
    idx = line.split(',')[0]
    if idx in dic:
        fo.write(dic[idx])
    else:
        fo.write(line)
        c+=1
print(c)
fo.close()
f.close()
