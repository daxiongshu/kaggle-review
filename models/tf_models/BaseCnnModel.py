import tensorflow as tf
import numpy as np
import pandas as pd
import time
from models.tf_models.BaseModel import BaseModel
from PIL import Image

class BaseCnnModel(BaseModel):
    def __init__(self,flags):
        super().__init__(flags)
        tf.GraphKeys.FEATURE_MAPS = 'feature_maps'

    def _build(self,inputs):
        # build the self.pred tensor
        raise NotImplementedError()


    def predict_lastconv(self,inputs, labels=None, activation="softmax"):

        # This function could be overwritten
        # default implementation is for multi classification
        with open(self.flags.pred_path,'w') as f:
            pass
        print()
        self._build(inputs)
        print()
        lastconv = tf.transpose(self.bottleneck, [0, 3, 1, 2])
        with tf.Session() as sess:
            self.sess = sess
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            if self.flags.log_path and self.flags.visualize is not None:
                summary_writer = tf.summary.FileWriter(self.flags.log_path, sess.graph)
            prev = 0
            count = 0
            try:
                while not coord.should_stop():
                    pred = sess.run([lastconv],feed_dict=self.feed_dict)[0]
                    n,c,h,w = pred.shape
                    m = h*w*c
                    pred = pred.reshape([n,m])
                    count+=pred.shape[0]
                    current = count//1000
                    if current>prev:
                        duration = time.time() - start_time
                        print ("%d samples predicted, Time %.3f"%(count, duration))
                        prev = current
                    with open(self.flags.pred_path,'a') as f:
                        pd.DataFrame(pred).to_csv(f, header=False,index=False, float_format='%.5f')
            except tf.errors.OutOfRangeError:
                duration = time.time() - start_time
                print ("Total time: %.3f"%duration)
            finally:
                coord.request_stop()
            coord.join(threads)


    def _conv2D(self, x, ksize, in_channel, out_channel, strides, layer_name, padding='SAME', activation=None, L2 = 1, use_bias=True):
        show_weight = self.flags.visualize and 'weight' in self.flags.visualize

        with tf.variable_scope(layer_name.split('/')[-1]):
            w,b = self._get_conv_weights(ksize,in_channel,out_channel,layer_name, use_bias=use_bias)
            net = tf.nn.conv2d(x,w,strides=strides,padding=padding)
            if use_bias:
                net = tf.nn.bias_add(net, b)

            net = self._activate(net, activation)
            if show_weight:
                tf.summary.histogram(name='W', values=w, collections=[tf.GraphKeys.WEIGHTS])
        return net


    def _max_pool2D(self, x, ksize, strides, padding, layer_name):
        with tf.name_scope(layer_name.split('/')[-1]):
            net = tf.nn.max_pool(x,ksize = ksize, strides = strides, padding = padding)
        return net

    def _get_conv_weights(self, ksize,in_channel,out_channel,layer_name, use_bias=True):
        if isinstance(ksize,list):
            kx,ky = ksize
        else:
            kx,ky = ksize,ksize

        w1 = self._get_variable(layer_name, name='weights', shape=[kx,ky,in_channel,out_channel])
        b1 = None
        if use_bias:
            b1 = self._get_variable(layer_name, name='bias', shape=[out_channel])
        return w1,b1

    def load_imagenet_names(self):
        self.imagenet_names = {}
        with open('data/imagenet_class.names') as f:
            for c,line in enumerate(f):
                self.imagenet_names[c] = line.strip()


    def conv_block(self, net, name, ksizes, filters, activations, strides, batchnorm=None, padding=None):
        assert len(filters) == len(ksizes)
        assert len(filters) == len(strides)
        assert len(filters) == len(activations)
        if batchnorm is None:
            batchnorm = [1 for i in filters]

        if padding is None:
            padding = ["SAME" for i in filters]

        if isinstance(padding,list)==False:
            padding = [padding for i in filters]

        if isinstance(strides[0],list) == False:
            strides = [[i,i] for i in strides]

        with tf.variable_scope(name.split('/')[-1]):
            for i in range(len(filters)):
                idx = i  if "resnet" not in name else i+1
                net = self._conv2D(net, ksize=ksizes[i], in_channel=net.get_shape().as_list()[-1],
                    out_channel=filters[i],
                    strides=[1,strides[i][0],strides[i][1],1], layer_name='%s/conv%d'%(name,idx), padding=padding[i],
                    activation = None)

                if batchnorm[i]:
                    net = self._batch_normalization(net, layer_name='%s/batch_norm%d'%(name,idx))
                
                net = self._activate(net, activations[i])

        return net


    def inference_one_image(self, imgname):
        img = Image.open(imgname).resize((224,224))
        img = np.array(img)
        img = np.expand_dims(img,axis=0).astype(float)
        inputs = tf.placeholder(tf.float32, shape=(None,224,224,3))
        self._build(inputs)
        self._get_summary()
        prob = tf.nn.softmax(self.logit)
        self.load_imagenet_names()

        with tf.Session() as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            start = time.time()
            if self.flags.log_path and self.summ_op is not None:
                summary_writer = tf.summary.FileWriter(self.flags.log_path, sess.graph)
                ypred,summary,step,x = sess.run([prob,self.summ_op,self.global_step,self.inputs], feed_dict={inputs: img})
                ypred = ypred[0]
                summary_writer.add_summary(summary, step)
            else:
                ypred = sess.run([prob],feed_dict={inputs: img})[0][0]
            print("time: %.3f"%(time.time()-start))
            print(ypred.shape, ypred[:10])
            yp = np.argmax(ypred)
            print(yp,self.imagenet_names[yp],ypred[yp])
            #self.print_all_variables()

    def just_graph(self):
        inputs = tf.placeholder(tf.float32, shape=(1,224,224,3))
        self.just_graph_with_input(inputs)

    def rgb_to_bgr(self, inputs):
        if True:
            if True:
                VGG_MEAN = [103.939, 116.779, 123.68]
                try:
                    red, green, blue = tf.split(inputs, 3, 3)
                except:
                    red, green, blue = tf.split(3,3,inputs)
                #assert red.get_shape().as_list()[1:] == [224, 224, 1]
                #assert green.get_shape().as_list()[1:] == [224, 224, 1]
                #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
                try:
                    bgr = tf.concat([
                        blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2]], axis=3)
                except:
                    bgr = tf.concat(3,[
                        blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2]])
        return bgr

    def predict_from_placeholder(self, activation=None):
        n,h,w,c = self.flags.batch_size,self.flags.height,self.flags.width,self.flags.color
        inputs = tf.placeholder(dtype=tf.float32,shape=[n,h,w,c])
        self._build(inputs)
        print()
        self.pred = self._activate(self.logit, activation)
        self._get_summary()

        with tf.Session() as sess:
            self.sess = sess
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if self.flags.log_path and self.flags.visualize is not None:
                summary_writer = tf.summary.FileWriter(self.flags.log_path, sess.graph)
            count = 0
            for batch in self._batch_gen():
                summary,pred = sess.run([self.summ_op, self.pred],feed_dict={inputs:batch})
                summary_writer.add_summary(summary, count)
                count+=1
                print("Batch",count)

    def _batch_gen(self):
        raise NotImplementedError()

    def dense_block(self, net, name, ksizes, filters, activations, strides, batchnorm=None, padding=None):
        assert len(filters) == len(ksizes)
        assert len(filters) == len(strides)
        assert len(filters) == len(activations)
        if batchnorm is None:
            batchnorm = [1 for i in filters]

        if padding is None:
            padding = ["SAME" for i in filters]

        if isinstance(padding,list)==False:
            padding = [padding for i in filters]

        if isinstance(strides[0],list) == False:
            strides = [[i,i] for i in strides]

        with tf.variable_scope(name.split('/')[-1]):
            inputs = [net]
            for i in range(len(filters)):
                out = 0
                for c,inx in enumerate(inputs):
                    net = self._conv2D(inx, ksize=ksizes[i], in_channel=inx.get_shape().as_list()[-1],
                        out_channel=filters[i],
                        strides=[1,strides[i][0],strides[i][1],1], layer_name='%s/conv%d_%d'%(name,i,c), padding=padding[i],
                        activation = None)
                    #out.append(net)
                    out = out+net
                #net = tf.add_n(out)
                net = out
                if batchnorm[i]:
                    net = self._batch_normalization(net, layer_name='%s/batch_norm%d'%(name,i))

                net = self._activate(net, activations[i])
                inputs.append(net)
        return net 

    def _scale(self, net, layer_name):
        with tf.variable_scope(layer_name.split('/')[-1]):
            w = self._get_variable(layer_name, name='w', shape=net.get_shape().as_list()[-1:])
            b = self._get_variable(layer_name, name='b', shape=net.get_shape().as_list()[-1:])
            return net*w+b


