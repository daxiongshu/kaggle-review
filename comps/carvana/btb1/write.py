import sys
import os

def write_all(path,head,name):
    files = ["%s/%s"%(path,i) for i in os.listdir(path) if i.endswith('.csv')]
    fo = open(name,'w')
    fo.write(head+'\n')
    for c,fx in enumerate(files):
        f = open(fx)
        for line in f:
            fo.write(line)
        f.close()
        if c%10 == 0:
            print(c)
    fo.close()

if __name__ == "__main__":
    path,head,name = sys.argv[1:]
    write_all(path,head,name)
