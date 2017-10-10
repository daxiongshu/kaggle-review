import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def distribution(data,xlabel="data",ylabel="percentage"):
    ax = plt.axes()
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ds = sns.distplot(data,ax=ax)
    plt.show()

def corr_heatmap(df,cols=None):
    sns.set()
    if cols is None:
        cols = [i for i in df.columns.values if df[i].dtype!='object']
    df = df[cols].corr()
    print(df.shape)
    sns.heatmap(df, annot=False)
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    corr_heatmap(pd.read_csv('../input/train.csv'))
