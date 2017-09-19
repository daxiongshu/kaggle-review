import os
import csv
import pickle

def run(flags):
    if flags.task == "prepare_cv_coord":
        prepare_cv_coord(flags)
    elif flags.task == "prepare_cv_hash":
        prepare_cv_hash(flags)
    elif flags.task == "prepare_sub_hash":
        prepare_sub_hash(flags)
    elif flags.task == "prepare_sub_coord":
        prepare_sub_coord(flags)
    elif flags.task == "eval_coord":
        eval_coord(flags)
    elif flags.task == "eval_hash":
        eval_hash(flags)
 
    else:
        from comps.mobike.sol_carl.xgb import train_predict
        path = "comps/mobike/sol_carl"
        params = {"objective": "rank:pairwise",
          "booster": "gbtree",
          "eval_metric": "map@3-",
          "eta": 0.1,
          "max_depth": 10,
          "silent": 1,
          "num_round": 116,
          "subsample":0.8,
          "colsample_bytree":0.8,
          "early_stopping_rounds":None
          }

        if flags.task == "cv_coord":
            params["num_round"] = 1000
            params["early_stopping_rounds"] = 10
            train_predict(trains = ["%s/data/%s"%(path,i) for i in ['tr_coord13-16.csv']],
                samples = ["%s/data/%s"%(path,i) for i in ['tr_coord18-19.csv']],
                tests=["%s/data/%s"%(path,i) for i in ['tr_coord18-19.csv']],out="%s/data/cv_coord_18-19.csv"%path,
                obj = "pairwise",params=params)

            params["num_round"] = 120
            params["early_stopping_rounds"] = None

            train_predict(trains = ["%s/data/%s"%(path,i) for i in ['tr_coord13-19.csv']],
                samples = ["%s/data/%s"%(path,i) for i in ['va_coord20-24.csv']],
                tests=["%s/data/%s"%(path,i) for i in ['va_coord20-24.csv']],out="%s/data/cv_coord_20-24.csv"%path,
                obj = "pairwise",params=params)



        if flags.task == "sub_coord":
            path = flags.input_path
            train_predict(trains = ["%s/%s"%(path,i) for i in ['train_coord13-24.csv']],
                samples = None,
                tests=["%s/%s"%(path,i) for i in ['test_coord.csv']],out="comps/mobike/sol_carl/data/sub_coord.csv",
                obj = "pairwise",params=params)

        path = "comps/mobike/sol_carl"
        if flags.task == "cv_hash":
            params["num_round"] = 1000 
            params["early_stopping_rounds"] = 10
            params["max_depth"] = 10
            params["subsample"] = 0.8 
            params["colsample_bytree"] = 0.8
            train_predict(trains = ["%s/data/%s.csv"%(path,i) for i in ['tr_norm_count','tr_distance']],
                samples = ["%s/data/%s_sample.csv"%(path,i) for i in ['va_norm_count','va_distance']],
                tests=["%s/data/%s.csv"%(path,i) for i in ['va_norm_count','va_distance']],out="%s/cv.csv"%path,
                obj = "pairwise",params=params)

        if flags.task == "sub_hash":
            params["num_round"] = 168
            params["max_depth"] = 10

            train_predict(trains = ["%s/data/%s.csv"%(path,i) for i in ['train_norm_count','train_distance']],
                samples = None, 
                tests=["%s/data/%s.csv"%(path,i) for i in ['test_norm_count','test_distance']],out="%s/sub.csv"%path,
                obj = "pairwise",params=params)

def prepare_cv_coord(flags):
    from comps.mobike.sol_carl.coord import build_hash_to_coord
    path = flags.input_path
    build_hash_to_coord(["%s/train.csv"%path,"%s/test.csv"%path])  

    path = "comps/mobike/sol_carl"    
    from comps.mobike.sol_carl.sample_data import sample_coord_data
    c,sc,dc = sample_coord_data(data="%s/data/tr_sort.csv"%path,start_day=13,end_day=16,
        out="%s/data/tr_coord13-16.csv"%path, min_distance=4000,
        topk=5,is_train=True)

    sample_coord_data(data="%s/data/tr_sort.csv"%path,start_day=18,end_day=19,
        out="%s/data/tr_coord18-19.csv"%path, min_distance=4000,
        topk=5,is_train=False,counter = c,scounter = sc,dist_dic=dc)

    del c,sc,dc

    c,sc,dc = sample_coord_data(data="%s/data/tr_sort.csv"%path,start_day=13,end_day=19,
        out="%s/data/tr_coord13-19.csv"%path, min_distance=4000,
        topk=5,is_train=True)

    sample_coord_data(data="%s/data/va_sort.csv"%path,start_day=20,end_day=24,
        out="%s/data/va_coord20-24.csv"%path, min_distance=4000,
        topk=5,is_train=False,counter = c,scounter = sc,dist_dic=dc)

def prepare_cv_hash(flags):
    path = "comps/mobike/sol_carl"
    from comps.mobike.sol_carl.sample_data import sample_hash_data
    counter,scounter,xc,xsc = sample_hash_data('%s/data/tr_sort.csv'%path,
        '%s/data/tr_norm_count.csv'%path,'%s/data/cv_coord_18-19.csv'%path,startday=17,
        threshold=10,max_loc=40,isva=0)
    sample_hash_data('%s/data/va_sort.csv'%path,'%s/data/va_norm_count.csv'%path,
        '%s/data/cv_coord_20-24.csv'%path,counter,scounter,xc=xc,xsc=xsc,
        threshold=10,max_loc=40,isva=1)

def prepare_sub_hash(flags):
    path = "comps/mobike/sol_carl"
    inpath = flags.input_path
    from comps.mobike.sol_carl.sample_data import sample_hash_data
    counter,scounter,xc,xsc = sample_hash_data('%s/train_sort.csv'%inpath,
        '%s/data/train_norm_count.csv'%path,'%s/data/cv_coord_20-24.csv'%path,startday=19,
        threshold=10,max_loc=40,isva=0)
    sample_hash_data('%s/test_sort.csv'%inpath,'%s/data/test_norm_count.csv'%path,
        '%s/data/sub_coord.csv'%path,counter,scounter,xc=xc,xsc=xsc,
        threshold=10,max_loc=40,isva=1)


def prepare_sub_coord(flags):
    path = flags.input_path
    from comps.mobike.sol_carl.sample_data import sample_coord_data
    c,sc,dc = sample_coord_data(data="%s/train_sort.csv"%path,start_day=13,end_day=24,
        out="%s/train_coord13-24.csv"%path, min_distance=4000,
        topk=5,is_train=True)

    sample_coord_data(data="%s/test_sort.csv"%path,start_day=0,end_day=1000,
        out="%s/test_coord.csv"%path, min_distance=4000,
        topk=5,is_train=False,counter = c,scounter = sc,dist_dic=dc)


def eval_coord(flags):
    path = "../input"
    dname = "%s/train_sort.csv"%path
    lname = "comps/mobike/sol_carl/data/train_coord_label.csv"
    get_label(dname,lname)

    path = "comps/mobike/sol_carl"
    from comps.mobike.sol_carl.evalx import eval
    eval(None,data=lname,sub="%s/data/cv_coord_18-19_sub.csv"%path,label=None,idx="orderid",candidate="true_coord",k=1)    
    eval(None,data=lname,sub="%s/data/cv_coord_20-24_sub.csv"%path,label=None,idx="orderid",candidate="true_coord",k=3)

def eval_hash(flags):
    from comps.mobike.sol_carl.evalx import eval
    path = "comps/mobike/sol_carl"
    eval(bdata="%s/data/tr_sort.csv"%path,data="%s/data/va_label.csv"%path,sub="%s/%s"%(path,flags.pred_path),label=None,idx="orderid",candidate="geohashed_end_loc",k=3,out="%s/%s.score"%(path,flags.pred_path))

def get_label(dname,lname):
    path = "comps/mobike/sol_carl"
    if os.path.exists(lname):
        return
    h2c = pickle.load(open("%s/data/h2c.p"%path))
    fo = open(lname,'w')
    fo.write("orderid,true_coord\n")
    for row in csv.DictReader(open(dname)):
        h = row['geohashed_end_loc']
        lat,lon = h2c[h]
        coord = "%s_%s"%(lat,lon)
        fo.write("%s,%s\n"%(row['orderid'],coord))
    fo.close()

 
