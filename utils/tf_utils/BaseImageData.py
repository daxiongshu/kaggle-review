import tensorflow as tf
import pandas as pd
import numpy as np
from utils.tf_utils.BaseData import BaseData
from PIL import Image
import os
from utils.tf_utils.utils import _int64_feature, _bytes_feature

LOG_EVERY = 1000

class BaseImageData(BaseData):

    def write_tfrecord(self, img_list, label_list, record_path):
        # write a single tfrecord
        if os.path.exists(record_path):
            print ("%s exists!"%record_path)
            return

        self._check_list()
        print ("write %s"%record_path)
        self._write_info()

        writer = tf.python_io.TFRecordWriter(record_path)
        c = 0
        for imgname,label in zip(img_list,label_list):

            img = Image.open(imgname).resize((self.flags.width, self.flags.height))
            data = np.array(img).astype(np.uint8)
            img,data = self._check_color(img,data)

            example = self._get_example(data,label)
            writer.write(example.SerializeToString())
            c+=1
            if c%LOG_EVERY == 0:
                print ("%d images written to tfrecord"%c)
        writer.close()
        print("writing %s done"%record_path)
          
    def _get_example(self,data,label):
        # by default, label is a single scaler per image
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image': _bytes_feature(data.tostring())}))
        return example

    def _write_info(self):
        # info: [n h w c classes]
        info = self.flags.record_path.replace('tfrecord','info')
        if os.path.exists(info):
            return
        num_imgs = len(os.listdir(self.flags.input_path))
        with open(info,'w') as fo:
            fo.write('%d %d %d %d %d\n'%(num_imgs,
                self.flags.height,self.flags.width,self.flags.color,
                self.flags.classes))

    def _check_list(self):

        if self.flags.height is None or self.flags.width is None:
            print ("width or height missing for writing tfrecords")
            assert False

    def _check_color(self,img,data):
        if self.flags.color==1:
            if len(data.shape) == 3:
                img = img.convert('1')
                data = np.array(data)
            elif len(data.shape) != 1:
                print("required color 1 image color %d"%len(data.shape))
                assert False
        elif self.flags.color == 3:
            if len(data.shape) != 3:
                print("required color 3 image color %d"%len(data.shape))
                assert False
        else:
            print("Unknown image color %d"%len(data.shape))
            assert False

        return img,data

    def _read_and_decode_single_example(self, filename_queue):
        features, label = self._parse(filename_queue)
        img,label = self._preprocess(features,label)
        img,label = self._augment(img,label)
        return img,label

    def _parse(self, filename_queue):
        with tf.name_scope("parsing"):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,
                features={'image':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64)
                }
            )
            label = tf.cast(features['label'],tf.int32)
        return features, label

    def _preprocess(self,features,label):
        width,height,color = self.flags.width, self.flags.height, self.flags.color
        with tf.name_scope("preprocessing"):
            img = tf.decode_raw(features["image"],tf.uint8)
            img.set_shape([width*height*color])
            img = tf.cast(img,tf.float32) #*(1.0/255)-0.5
            img = tf.reshape(img,[height,width,color])
        return img,label

    def _augment(self,img,label):
        # this function is to be overwritten
        if self.flags.augmentation:
            with tf.name_scope("augmentation"):
                img = img
        return img,label

    def batch_generator_train(self, is_onehot):
        epochs = self.flags.epochs
        batch_size = self.flags.batch_size

        self.write_tfrecords()
        files = self._get_tfrecord_paths()
        print(files)

        with tf.name_scope("Input"):
            filename_queue = tf.train.string_input_producer(files, num_epochs=epochs, shuffle = True)
            img,label = self._read_and_decode_single_example(filename_queue)
            imgs,labels = self._batching(img,label)
        samples = self._get_num_samples() 
        return imgs,labels,samples

    def batch_generator_predict(self, is_onehot=False):
        files = self._get_tfrecord_paths()
        print(files)
        with tf.name_scope("Input"):
            filename_queue = tf.train.string_input_producer(files,num_epochs=1)
            img,label = self._read_and_decode_single_example(filename_queue)
            imgs,labels = self._batching(img,label)
        return imgs, labels

    def _get_tfrecord_paths(self):
        # files are determined by flags.task
        raise NotImplementedError()

    def _get_num_samples(self):
        raise NotImplementedError()

