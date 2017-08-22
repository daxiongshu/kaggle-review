from time import time
_start = time()
try:
    import psutil
    _summ = psutil.virtual_memory()
except:
    pass

def print_mem_time(tag):
    print(tag," time %.2f seconds"%(time()-_start), end='')
    try:
        summ = psutil.virtual_memory()
        print(" Used: %.2f GB %.2f%%"%((summ.used-_summ.used)/(1024.0*1024*1024),summ.percent-_summ.percent))
    except:
        print()

