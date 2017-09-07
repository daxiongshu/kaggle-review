import pandas as pd

def rm_const_cols(df,bar=0.999):
    cols = df.columns.values
    const = []
    for col in cols:
        tmp = df[col].value_counts()    
        ratio = tmp.max()/tmp.sum()
        if ratio>=bar:
            const.append(col)   
            print(ratio,col)
    print("df shape",df.shape,"num const cols",len(const))
    if len(const)==0:
        return df
    else:
        return df.drop(const,axis=1) 
        
