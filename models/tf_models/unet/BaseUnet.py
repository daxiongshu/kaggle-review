#based on https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model 
from models.tf_models.BaseCnnModel import BaseCnnModel
import tensorflow as tf
import numpy as np
from utils.image_utils.cv_util import random_batch_generator
import time
from utils.utils import print_mem_time
import pandas as pd

class BaseUnet(BaseCnnModel):

    def __init__(self,flags,DATA=None):
        super().__init__(flags)
        self.keep_prob = self.flags.keep_prob
        self.DATA = DATA
        self.epoch = 0

    # predict with placeholders
    def predictPL(self):
        B = self.flags.batch_size
        W,H,C = self.flags.width, self.flags.height, self.flags.color
        inputs = tf.placeholder(dtype=tf.float32,shape=[None,H,W,C])

        #with open(self.flags.pred_path,'w') as f:
        #    pass

        self._build(inputs,resize=False)
        counter = 0
        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for imgs,imgnames in self.DATA.test_generator():
                pred = sess.run(self.logit,feed_dict={inputs:imgs})
                np.save("%s/%d.npy"%(self.flags.pred_path,counter),{"pred":pred,"name":imgnames})
                counter+=len(imgs)
                if counter/B%10 ==0:
                    print_mem_time("%d images predicted"%counter)

    # train with placeholders
    def trainPL(self, mode="random_data", do_va=False):

        B = self.flags.batch_size
        W,H,C = self.flags.width, self.flags.height, self.flags.color
        inputs = tf.placeholder(dtype=tf.float32,shape=[B,H,W,C])
        labels = tf.placeholder(dtype=tf.float32,shape=[B,H,W])
        
        self._build(inputs,resize=False)
        self._get_loss(labels)
        self._get_opt()
        self._get_summary()
        tr_acc,va_acc,counter,last,vacc,va_summary = 0,0,0,0,0,None
        tr_summary,va_summary = None,None

        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self._setup_writer()
            for imgs,masks,epoch in self._tr_generator(mode):
                if self.flags.log_path and self.summ_op is not None and counter%10 == 0:
                    # only learning curve for train
                    _, loss, acc,tr_summary = sess.run([self.opt_op,self.loss,self.acc,self.scaler_op],
                        feed_dict={inputs:imgs, labels:masks})
                else:
                    _, loss, acc = sess.run([self.opt_op,self.loss,self.acc],
                        feed_dict={inputs:imgs, labels:masks})

                if do_va and counter%10 == 0:
                    _,vacc,va_summary = self._validation(inputs,labels,counter)
                    self._run_writer(tr_summary,va_summary,counter/10)
                tr_acc,va_acc,counter,last = self._print_and_save(acc,vacc,tr_acc,va_acc,
                    counter,last,epoch)
            self.epoch = self.flags.epochs
            self._save()



    def _setup_writer(self):
        if self.flags.log_path and self.summ_op is not None:
            path = "%s_train"%self.flags.log_path
            self.train_summary_writer = tf.summary.FileWriter(path, self.sess.graph)
            path = "%s_valid"%self.flags.log_path
            self.test_summary_writer = tf.summary.FileWriter(path, self.sess.graph)

    def _run_writer(self,tr_summary,va_summary,step):
        if self.flags.log_path and self.summ_op is not None and va_summary:
            self.train_summary_writer.add_summary(tr_summary, step)
            self.test_summary_writer.add_summary(va_summary, step)

    def _print_and_save(self,acc,vacc,tr_acc,va_acc,counter,last,epoch): 
        tr_acc = self._get_acc_loss(tr_acc,acc,ratio=0.99)
        B = self.flags.batch_size
        counter += 1
        if counter==1:
            print("\nFirst Train %s %.3f"%(self.flags.metric,tr_acc))
        if counter%10 == 0:
            if vacc==0:
                line = "Epoch %d Samples %d Train %s %.4f"%(epoch,counter*B,self.flags.metric,tr_acc)
            else:
                va_acc = self._get_acc_loss(va_acc,vacc)
                line = "Epoch %d Samples %d Train %s %.4f Valid %s %.4f"%(epoch,counter*B,self.flags.metric,tr_acc,self.flags.metric,va_acc)

            print_mem_time(line)
        if self.flags.epochs is None:
            if counter%100 == 0:
                self.epoch = counter
                self._save()
        else:
            if epoch>last:
                last = epoch
                self.epoch = epoch
                self._save()
        return tr_acc,va_acc,counter,last

    def _validation(self,inputs,labels,counter):
        if self.DATA is None:
            return None,None,None
        va_summary = None
        for imgs,masks,_ in self.DATA.va_generator(first=counter==0):

            if self.flags.log_path and self.summ_op is not None:
                # full visualization for validation
                loss, acc,va_summary,logit = self.sess.run([self.loss,self.acc,self.summ_op,self.logit],
                    feed_dict={inputs:imgs, labels:masks})
            else:
                loss, acc = self.sess.run([self.loss,self.acc],
                    feed_dict={inputs:imgs, labels:masks})
            break # only run one batch
        return loss,acc,va_summary

    def _tr_generator(self, mode):
        B,W,H = self.flags.batch_size, self.flags.width, self.flags.height
        if mode == "random_data":
            return random_batch_generator(B,W,H)
        elif self.DATA is not None:
            return self.DATA.tr_generator()
        else:
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

        with tf.name_scope("summary"):
            if self.flags.visualize:
                tf.summary.scalar(name='dice coef', tensor=self.acc, collections=[tf.GraphKeys.SCALARS])

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
