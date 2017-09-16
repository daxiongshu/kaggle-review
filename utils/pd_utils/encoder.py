from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def lbl_encode(df_tr,df_te=None,cols=None,objonly=True):
    print("label encode ...")
    lbl = LabelEncoder()
    if df_te is not None:
        df = df_tr.append(df_te)
        if cols is None:
            cols = set(df_tr.columns.values).intersection(set(df_te.columns.values))
    else:
        df = df_tr
        if cols is None:
            cols = df_tr.columns.values
    encoded = []
    for col in cols:
        if objonly and df[col].dtype!='object':
            continue
        encoded.append(col)
        lbl.fit(df[col].map(str))
        df_tr[col] = lbl.transform(df_tr[col].map(str))
        if df_te is not None:
            df_te[col] = lbl.transform(df_te[col].map(str))
    print('lbl encode:',encoded)

