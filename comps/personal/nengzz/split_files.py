f = open('../input/training_text')
f.readline()
for line in f:
    xx = line.split('||')
    fo = open('comps/personal/nengzz/data/%s.txt'%xx[0],'w')
    fo.write(xx[1])
    fo.close()
f.close()
