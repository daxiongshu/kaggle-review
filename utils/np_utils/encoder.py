import numpy as np

def onehot_encode(y,n=None):
    """
    Input: 
        y 1d array of lenth B, elements from {0,1,..,n-1}
        n: the number of classes
    Return: yp 2d aray [B,N]
    """
    if n is None:
        n = np.max(y)+1
    yp = np.zeros([y.shape[0],n])
    x = np.arange(y.shape[0])
    yp[x,y] = 1
    return yp

if __name__ == "__main__":
    y = np.random.randint(0,10,5)
    yp=onehot_encode(y,10)
    print(y)
    print(yp)
