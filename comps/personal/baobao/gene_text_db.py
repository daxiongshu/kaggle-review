from comps.personal.personal_db import personalDB

class geneTextDB(personalDB):
    def __init__(self,flags,W,bar,tag='gene',files="all"):
        name = "%s_%d_%d"%(tag,W,bar)
        super().__init__(flags,name=name,build=False)
        fnames = "training_text,test_text_filter,stage2_test_text.csv".split(',')
        self.names = [i.split('.')[0] for i in fnames]
        self.fnames = ["%s_%s"%(name,fname) for fname in fnames]
        self.path = flags.data_path
        self._build(flags,files)

    def poke(self):
        for name,data in self.data.items():
            if 'text' not in name:
                continue
            #data['Text'] = data['Text'].apply(lambda x: str(x).encode('utf-8'))#.encode('utf-8')
            #print(name,data.head())
            #data['lw'] = data['Text'].apply(lambda x: len(x.strip().split()))
            data['lw'] = data['Text']!="__NULL__"
            print(name,data.shape,data['lw'].mean())
