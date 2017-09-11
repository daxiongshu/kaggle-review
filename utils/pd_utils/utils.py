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
    if len(const)>0:
        df.drop(const,axis=1,inplace=True)

def get_ymd(df,col,deli='-',order="ymd"):
    def _parse_order(order):
        return {i:c for c,i in enumerate(order)}
    order_dic = _parse_order(order) 
    df["year"] = df[col].apply(lambda x: x.split(deli)[order_dic['y']]).astype(int)
    df["month"] = df[col].apply(lambda x: x.split(deli)[order_dic['m']]).astype(int)
    df["day"] = df[col].apply(lambda x: x.split(deli)[order_dic['d']]).astype(int)

