import os
import re
import csv

def poke_gene_texts(flags):
    import pandas as pd
    path = flags.input_path
    train = False
    if train:
        data = pd.read_csv("%s/training_text"%path,header=None,sep="\|\|",skiprows=1,names=['ID','Text'])
        datav = pd.read_csv("%s/training_variants"%path)
    else:
        data = pd.read_csv("%s/stage2_test_text.csv"%path,header=None,sep="\|\|",skiprows=1,names=['ID','Text'])
        datav = pd.read_csv("%s/stage2_test_variants.csv"%path)
        datav['gv'] = datav['Gene']+'-'+datav['Variation']
        data2 = pd.read_csv("%s/test_text"%path,header=None,sep="\|\|",skiprows=1,names=['ID','Text'])
        datav2 = pd.read_csv("%s/test_variants"%path)
        datav2['gv'] = datav2['Gene']+'-'+datav2['Variation']
        mask = datav['gv'].isin(datav2['gv'])
        datav = datav[~mask]
        data = data[~mask]
    data['Gene'] = datav['Gene'].values
    data['Gene'] = data['Gene'].apply(lambda x: gene_map(x))
    data['Gene'] = data['Gene'].apply(lambda x: re.findall(r'[a-zA-Z]+',x)[0])
    #data['Class'] = datav['Class'].values
    data['gin'] = data.apply(lambda r: len(re.findall(r['Gene'].lower(),r['Text'].lower()))>0, axis=1)
    mask = data['gin']==0
    #print(data[mask]['Class'].value_counts())
    print(data['gin'].mean())
    print(data[mask]['Gene'].unique())
    print(data[mask]['Gene'].unique().shape)
    print(data[mask]['ID'][:10])

def write_gene_text(flags,window=10,bar=0):
    import pandas as pd
    path = flags.input_path
    opath = flags.data_path
    W = window
    for name in ['training_text','test_text_filter','stage2_test_text.csv']:
        oname = "%s/gene_%s_%d_%d"%(opath,name,W,bar)
        if os.path.exists(oname):
            continue
        s = pd.read_csv("%s/%s"%(path,name),header=None,sep="\|\|",skiprows=1,names=['ID','Text'])
        vname = name.replace('text','variants')
        sv = pd.read_csv("%s/%s"%(path,vname))
        s['Gene'] = sv['Gene'].values
        s['GText'] = s.apply(lambda row:find_gene_text(row['Text'],row['Gene'],W,bar),axis=1)
        fo = open(oname,'w')
        fo.write("ID,Text\n")
        for i in range(s.shape[0]):
            fo.write("%d||%s\n"%(s.loc[i,'ID'],s.loc[i,'GText']))
        fo.close()

def write_gene_text_pypy(flags,tag='Gene',window=10,bar=0):
    path = flags.input_path
    opath = flags.data_path
    W = window
    for name in ['training_text','test_text_filter','stage2_test_text.csv']:
        oname = "%s/%s_%d_%d_%s"%(opath,tag.lower(),W,bar,name)
        if os.path.exists(oname):
            continue
        f1 = open("%s/%s"%(path,name))
        vname = name.replace('text','variants')
        f2 = csv.DictReader(open("%s/%s"%(path,vname)))
        f1.readline()
        fo = open(oname,'w')
        fo.write("ID,Text\n")
        for c,l1 in enumerate(f1):
            if c%1000 == 0:
                print(c,oname)
            l2 = f2.next()
            #print(l2)
            ID,gene = l2['ID'],l2[tag]
            text = l1.strip().split('||')[1]
            gtext = find_gene_text(text,gene,W,bar,isvar=tag=='Variation')
            fo.write("%s||%s\n"%(ID,gtext))
        fo.close()
        f1.close()
        print(oname,'done')

def find_gene_text(s,gene,W,bar,isvar=False):
    s,gene = s.upper(),gene.upper()
    genes = gene_map(gene,islist=True,isvar=isvar)
    wins = []
    for gene in genes:
        wins.extend(find_gene_words(s,gene,W))
        wins = merge_window(wins)
        if len_window(wins)>bar:
            break
    return get_words_from_window(s,wins)

def find_gene_words(s,gene,w):
    s,gene = s.upper(),gene.upper()
    p = re.compile(gene)
    result = []
    for m in p.finditer(s):
        pos = m.start()
        l,r = find_window(s,pos,w)
        result.append((l,r))
    result = merge_window(result)
    return result

def get_words_from_window(s,wins):
    result =''
    for win in wins:
        l,r = win
        result = result + ' ' + s[l:r].strip()
    result = result.strip().lower()
    if result == '':
        result = "__NULL__"
    return result

def len_window(x):
    s = 0
    for i,j in x:
        s+=(j-i)
    return s

def find_window(s,pos,w):
    l = max(0,pos-w)
    while l>0 and s[l]!=' ':
        l-=1
    r = min(len(s)-1,pos+w)
    while r<len(s)-1 and s[r]!=' ':
        r+=1
    return l,r

def merge_window(windows):
    result = []
    for w in windows:
        if len(result)==0:
            result.append(w)
            continue
        l1,r1 = w
        bad = []
        for c,nw in enumerate(result):
            l2,r2 = nw
            if max(l1,l2)<min(r1,r2):
                l1 = min(l1,l2)
                r1 = max(r1,r2)
                bad.append(c)
        result = [i for c,i in enumerate(result) if c not in bad]
        result.append((l1,r1))
    return result

def check_merge():
    x = [(0,24),(10,15),(20,30),(50,60)]
    result = merge_window(x)
    print(x,result)

def gene_map(gene,islist=False,isvar=False):
    result = [gene]
    if isvar:
        return result
    if len(re.findall("TP53",gene)):
        result.append("P53")
    elif len(re.findall("DICER",gene)):
        result.append("DICER")
    elif len(re.findall("CCND",gene)):
        result.append("cyclin D")
    elif len(re.findall("TGFBR",gene)):
        result.append("TGF")
    elif len(re.findall("FBXW",gene)):
        result.append("FB")
    if islist:
        result.extend(['loss of function','loss-of-function',
            'gain of function','gain-of-function','lof','gof'])
        return result
    else:
        return result[-1]
