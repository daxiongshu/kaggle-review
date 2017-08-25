from utils.utils import parallel_run
import numpy as np
import os
from PIL import Image
from scipy.misc import imresize

# thanks to https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of the 
    # original image) by setting those pixels to '0' explicitly. We do not 
    # expect these to be non-zero for an accurate mask, so should not 
    # harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def post_sub_one(inx):
    w,h = 1918,1280
    path,out,threshold = inx
    data = np.load(path).item()
    imgs,pred = data['name'], data['pred']
    #print(pred.shape)
    fo = open(out,'w')
    #masks = pred>threshold
    for name,mask in zip(imgs,np.squeeze(pred)):
        mask = imresize(mask,[h,w])
        mask = mask>threshold
        code = rle_encode(mask)
        code = [str(i) for i in code]
        code = " ".join(code)
        fo.write("%s,%s\n"%(name,code))
    fo.close()
    return 0

def post_sub_all(path,threshold):
    paths = ["%s/%s"%(path,i) for i in os.listdir(path) if i.endswith('npy')]
    args = [[i,i.replace('npy','csv'),threshold ] for i in paths]
    parallel_run(post_sub_one,args)

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


