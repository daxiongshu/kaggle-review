import pandas as pd
s1 = pd.read_csv('comps/personal/baobao/backup/91_87_86_84_73.9/sub_stage2.csv',index_col='ID')
s2 = pd.read_csv('comps/personal/baobao/backup/91_88_87_86_74/sub_stage2.csv',index_col='ID')
for i in range(1,10):
    s1['class%d'%i] = (s1['class%d'%i]+s2['class%d'%i])/2
s1.to_csv('nave2.csv',index=True)
