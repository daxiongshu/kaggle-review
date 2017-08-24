#based on https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model 
from models.tf_models.BaseCnnModel import BaseCnnModel
import tensorflow as tf
import numpy as np
from utils.image_utils.cv_util import random_batch_generator
import time
from utils.utils import print_mem_time

class BaseUnet(BaseCnnModel):

    def __init__(self,flags):
        super().__init__(flags)
        self.keep_prob = self.flags.keep_prob

    def train(self,mode):
        B = self.flags.batch_size

        W,H,C = self.flags.width, self.flags.height, self.flags.color
        #W,H,C = 224,224,3
        inputs = tf.placeholder(dtype=tf.float32,shape=[B,H,W,C])
        # NHWC of tensorflow https://www.tensorflow.org/performance/performance_guide
        labels = tf.placeholder(dtype=tf.float32,shape=[B,H,W])

        self._build(inputs,resize=False)
        self._get_loss(labels)
        self._get_opt()
        self._get_summary()
        acc_loss = 0
        acc_c = 0
        counter = 0
        with tf.Session() as sess:
        #with tf.Session() as sess:
            self.sess = sess
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for imgs,masks in self._train_batch_generator(mode):
                _, loss, acc = sess.run([self.opt_op,self.loss,self.acc],
                    feed_dict={inputs:imgs, labels:masks})
                acc_loss = self._get_acc_loss(acc_loss,loss)
                acc_c = self._get_acc_loss(acc_c,acc)
                counter += 1
                if counter==1:
                    print("First loss %.3f"%acc_loss)
                if counter%10 == 0:
                    #print("Time %.2f s Samples %d Loss %.4f %s %.4f"%(time.time()-start_time,counter*B,acc_loss,self.flags.metric,acc_c))
                    line = "Samples %d Loss %.4f %s %.4f"%(counter*B,acc_loss,self.flags.metric,acc_c)
                    print_mem_time(line)
                if counter%100 == 0:
                    self.epoch = counter
                    self._save()


    def _train_batch_generator(self,mode):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        if mode == "random":
            return random_batch_generator(B,W,H)
        else:
            print("unknow mode",mode)
            assert False

    def _get_loss(self,labels):

        with tf.name_scope("Loss"):
            """
            with tf.name_scope("logloss"):
                logit = tf.squeeze(tf.nn.sigmoid(self.logit))
                self.loss = tf.reduce_mean(self._logloss(labels, logit))
            """
            with tf.name_scope("L2_loss"):
                if self.flags.lambdax:
                    lambdax = self.flags.lambdax
                else:
                    lambdax = 0
                self.l2loss = lambdax*tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            with tf.name_scope("dice_coef"):
                #yp_label = tf.cast(logit>self.flags.threshold, tf.float32)
                logit = tf.squeeze(self.logit)
                self.acc = tf.reduce_mean(self._dice_coef(labels,logit))
                self.metric = "dice_coef"
                self.loss = -self.acc

    def _dice_coef(self,y_true,y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2.0 * intersection + 1.0) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.0)

    def _jacard_coef(self,y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        return (intersection + 1.0) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.0)

    def _jacard_coef_loss(self, y_true, y_pred):
        return -self.jacard_coef(y_true, y_pred)

    def _dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)

    def _deconv2D(self, x, ksize, in_channel, out_channel, strides, layer_name, padding='SAME', activation=None, L2 = 1):
        show_weight = self.flags.visualize and 'weight' in self.flags.visualize

        with tf.variable_scope(layer_name.split('/')[-1]):
            w,b = self._get_conv_weights(ksize,in_channel,out_channel,layer_name)
            out_shape = [tf.shape(x)[0], tf.shape(x)[1]*strides[1], tf.shape(x)[2]*strides[2], out_channel]
            net = tf.nn.conv2d_transpose(x,w,tf.stack(out_shape),strides=strides,padding=padding)
            net = tf.nn.bias_add(net, b)

            net = self._activate(net, activation)
            if show_weight:
                tf.summary.histogram(name='W', values=w, collections=[tf.GraphKeys.WEIGHTS])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w)* L2)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b)* L2)
        return net

    def debug(self):
        inputs = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3])
        self._build(inputs)
        self.print_all_variables()
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if self.flags.log_path and self.flags.visualize is not None:
                summary_writer = tf.summary.FileWriter(self.flags.log_path, sess.graph)
