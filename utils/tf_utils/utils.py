import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
import os
import sys
import time
import csv

def bbox_iou(box1,box2):
    """
        input: box1 (l,r,t,b)
        output: iou
    """
    l1,r1,t1,b1 = box1
    l2,r2,t2,b2 = box2
    l = max(l1,l2)
    r = min(r1,r2)
    t = max(t1,t2)
    b = min(b1,b2)
    if l>r or t>b:
        return 0
    a0 = (r-l)*(t-b)
    a1 = (r1-l1)*(t1-b1)
    a2 = (r2-l2)*(t2-b2)
    aa = min(a1,a2)
    return a*1.0/aa

def intseqfea(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def floatseqfea(values):
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))

def read_embed_dic(name):
    """
        return a {}
               token -> id: hello -> 100
    """
    dic = pd.read_csv(name,index_col="token")
    return dic["idx"].to_dict()

def write_embed_dic(embed,dic):
    """
        write a csv file:
               token,id: hello,100
    """

    print("write %s"%dic)
    with open(dic,"w") as fo:
        fo.write("token,idx\n")
        with open(embed) as f:
            f.readline() # the first line is rows, cols
            for c,line in enumerate(f):
                x = line.find(' ')
                token = line[:x]                        
                fo.write('"%s",%d\n'%(token,c))
    print("write %s done"%dic)

def encode_line(line,dic):
    """
        encode each token to an integer id
    """
    xx = line.split()
    res = []
    for x in xx:
        res.append(str(dic.get(x,0)))
    return " ".join(res)

        
     
