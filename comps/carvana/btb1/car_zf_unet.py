from models.tf_models.unet.zf_unet import ZF_UNET
from utils.image_utils.cv_util import random_batch_generator
from utils.utils import split
from random import shuffle
from PIL import Image
import numpy as np

class carZF_UNET(ZF_UNET):

    def _train_batch_generator(self,mode):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        if mode == "random":
            return random_batch_generator(B,W,H)
        else:
            return self.car_train_batch_generator()

    def car_train_batch_generator(self,keep_data = True):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        fold = self.flags.fold
        epochs = self.flags.epochs
        dic = split(self.flags)
        imgs = sum([dic[i] for i in dic if i!=fold],[])
        del dic
        loaded = {}

        for i in range(epochs):
            shuffle(imgs)
            xs,ys = [],[]
            for img in imgs:
                if len(xs) == B:
                    yield np.array(xs),np.array(ys)
                    del xs,ys
                    xs,ys = [],[]

                if img in loaded:
                    x,y = loaded[img]
                else:
                    x = np.array(Image.open(img).resize([W,H]))
                    label = img.replace(".jpg","_mask.gif").replace("train","train_masks")
                    y = np.array(Image.open(label).resize([W,H]))
                    if keep_data:
                        loaded[img] = (x,y)
                xs.append(x)
                ys.append(y)

                

        
