import pandas as pd
import os
def sort_by_time(name):
    oname = name.replace('.csv','_sort.csv')
    if os.path.exists(oname):
        return
    s = pd.read_csv(name)
    print("%s loaded"%name)
    s = get_time(s)
    s = s.sort_values(by = 'time')
    s.to_csv(oname,index=False)
    print("sort %s done"%name)

def get_time(s):
    s['month'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[1]).astype(int)
    s['day'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[-1]).astype(int)
    s['year'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[0]).astype(int)
    s['hour'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[0]).astype(int)
    s['min'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[1]).astype(int)
    s['sec'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[-1].split('.')[0]).astype(int)
    s['time'] = ((((s.month-5)*30+s.day-9)*24+s.hour)*60+s['min'])*60 + s['sec']
    return s.drop('month,day,year,hour,min,sec'.split(','),axis=1)

if __name__ == "__main__":
    sort_by_time('comps/mobike/sol_carl/data/tr.csv')
    sort_by_time('comps/mobike/sol_carl/data/va.csv')
    sort_by_time('../input/train.csv')
    sort_by_time('../input/test.csv')
