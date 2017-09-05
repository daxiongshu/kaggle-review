import seaborn as sns
import matplotlib.pyplot as plt

def distribution(data,xlabel="data",ylabel="percentage"):
    ax = plt.axes()
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ds = sns.distplot(data,ax=ax)
    plt.show() 
