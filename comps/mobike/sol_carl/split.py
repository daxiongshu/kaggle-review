import pandas as pd
import os
def split():
    if os.path.exists('comps/mobike/sol_carl/data/va.csv'):
        return
    path = "../input/train.csv"
    s = pd.read_csv(path)
    s['day'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[-1]).astype(int)
    mask = s['day']>19
    s[mask].drop('geohashed_end_loc',axis=1).to_csv('comps/mobike/sol_carl/data/va.csv',index=False)
    s[~mask].to_csv('comps/mobike/sol_carl/data/tr.csv',index=False)
    s[mask][['orderid','geohashed_end_loc']].to_csv('comps/mobike/sol_carl/data/va_label.csv',index=False)

if __name__ == "__main__":
    split()
