from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def lbl_encode(df_tr,df_te,cols=None,objonly=True):
    lbl = LabelEncoder()
    if cols is None:
        cols = set(df_tr.columns.values).intersection(set(df_te.columns.values))
    df = df_tr.append(df_te)
    encoded = []
    for col in cols:
        if objonly and df[col].dtype!='object':
            continue
        encoded.append(col)
        lbl.fit(df[col].map(str))
        df_tr[col] = lbl.transform(df_tr[col].map(str))
        df_te[col] = lbl.transform(df_te[col].map(str))
    print(encoded)

