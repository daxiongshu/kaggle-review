import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def scatter(x,y,xlabel='x',ylabel='y',title=None,line=False,name=None,show=False):
    sns.set()
    title = "%s vs %s"%(xlabel,ylabel) if title is None else title
    plt.scatter(x,y)
    if line:
        plt.plot(x,y)
    plt.title(title)
    plt.ylabel('y: %s'%ylabel)
    plt.xlabel('x: %s'%xlabel)
    if name is not None:
        #fig = plt.Figure()
        plt.savefig(name)
    if show:
        plt.show()
    plt.clf()

def distribution(data,xlabel="data",ylabel="percentage",name=None):
    ax = plt.axes()
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ds = sns.distplot(data,ax=ax)
    plt.show()
    if name is not None:
        ds.get_figure().savefig(name)

def corr_heatmap(df,cols=None,name=None):
    sns.set()
    if cols is None:
        cols = [i for i in df.columns.values if df[i].dtype!='object']
    df = df[cols].corr()
    print(df.shape)
    ds = sns.heatmap(df, annot=False)
    plt.show()
    if name is not None:
        ds.get_figure().savefig(name)
    

if __name__ == "__main__":
    import pandas as pd
    #corr_heatmap(pd.read_csv('../input/train.csv'))
    scatter([1,2,3,4],[5,6,7,8],name='xx.png') 
