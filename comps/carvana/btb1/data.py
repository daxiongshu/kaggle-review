from utils.tf_utils.BaseImageData import BaseImageData
import os
import numpy as np
from random import shuffle
from PIL import Image
import tensorflow as tf
from utils.tf_utils.utils import _int64_feature, _bytes_feature
from utils.utils import split

class tfCarData(BaseImageData):

    def tr_generator(self,keep_data = True):
        fold = self.flags.fold
        epochs = self.flags.epochs
        dic = split(self.flags)
        imgs = sum([dic[i] for i in dic if i!=fold],[])
        folds = [i for i in dic if i!=fold]
        print("Train using folds {} images {}".format(folds,len(imgs)))
        del dic
        return self._gen_random_batch(epochs,imgs,keep_data)

    def va_generator(self,keep_data = True,first=False):
        fold = self.flags.fold
        dic = split(self.flags)
        imgs = dic[fold]
        del dic
        if first:
            print("Valid using fold {} images {}".format(fold,len(imgs)))
        return self._gen_random_batch(1,imgs,keep_data)

    def test_generator(self,keep_data=False):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        path = self.flags.input_path
        loaded = {}
        if "cv" not in self.flags.task:
            # test images to submit
            imgs = ["%s/%s"%(path,i) for i in os.listdir(path)]
        else:
            imgs = split(self.flags)[self.flags.fold]
        self.test_imgs = imgs
        xs,ns = [],[]
        for img in imgs:
            if len(xs) == B:
                yield np.array(xs),ns
                del xs,ns
                xs,ns = [],[]
            if img in loaded:
                x = loaded[img]
            else:
                x = np.array(Image.open(img).resize([W,H]))
                if keep_data:
                    loaded[img] = x
            xs.append(x)
            ns.append(img.split('/')[-1])
        if len(xs):
            yield np.array(xs),ns

    def _gen_random_batch(self,epochs,imgs,keep_data):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        loaded = {}
        xs,ys = [],[]
        for i in range(epochs):
            shuffle(imgs)
            for img in imgs:
                if len(xs) == B:
                    yield np.array(xs),np.array(ys),i
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

    def write_tfrecords(self):
        task = self.flags.task
        if "cv" in task:
            self.write_cv_tfrecords()
        else:
            self.write_test_tfrecords()

    def write_cv_tfrecords(self):
        folds = self.flags.folds
        dic = split(self.flags)
        for i in range(folds):
            record_path = self.flags.record_path.replace(".tfrecords","_%d.tfrecords"%i)
            imgs = dic[i]
            labels = [img.replace(".jpg","_mask.gif").replace("train","train_masks") for img in imgs]
            self.write_tfrecord(imgs, labels, record_path)

    def write_test_tfrecords(self):
        record_path = self.flags.record_path
        path = self.flags.input_path
        imgs = ["%s/%s"%(path,img) for img in os.listdir(path)]
        labels = None
        self.write_tfrecord(imgs, labels, record_path)

    def _get_example(self,data,label):
        # by default, label is a single scaler per image
        label = Image.open(label).resize((self.flags.width,self.flags.height))
        if label is not None:
            label = np.array(label).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(label.tostring()),
                'image': _bytes_feature(data.tostring())}))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(data.tostring())}))
        return example

    def _get_tfrecord_paths(self):
        task = self.flags.task
        folds = self.flags.folds
        fold = self.flags.fold
        record_path = self.flags.record_path
        if "cv_train" == task:
            return [record_path.replace(".tfrecords","_%d.tfrecords"%i) for i in range(folds) if i!= fold]
        elif "cv_predict" == task:
            return [record_path.replace(".tfrecords","_%d.tfrecords"%fold)]
        elif "test" == task:
            return [record_path]
        else:
            print("unknown task",task)
            assert False

    def _batching(self,x,y):
        """
            Input:
                x,y: [F1,F2,..] single tensors
            Return:
                xs,ys: [B,F1,F2..] batched tensors
        """
        pass
