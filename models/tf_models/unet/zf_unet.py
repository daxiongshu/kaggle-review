#based on https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model
from models.tf_models.unet.BaseUnet import BaseUnet
import tensorflow as tf

class ZF_UNET(BaseUnet):

    def _build(self,inputs,resize=True, dropout=False):
        net_name = "ZF_UNET"
        width,height = self.flags.width,self.flags.height
        with tf.variable_scope(net_name):

            if self.flags.visualize and 'image' in self.flags.visualize:
                tf.summary.image(name="images", tensor=inputs,
                    max_outputs=3, collections=[tf.GraphKeys.IMAGES])

            net = inputs
            if resize:
                with tf.name_scope("Resize"):
                    net = tf.image.resize_images(net,(height,width))

            with tf.name_scope("Preprocess"):
                net = net/256.0 - 0.5

            conv1, net = self._ZF_down_block(net, ksizes=[3,3], filters=[32,32], 
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,1), keep_prob = self.keep_prob)

            conv2, net = self._ZF_down_block(net, ksizes=[3,3], filters=[64,64],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,2), keep_prob = self.keep_prob)    

            conv3, net = self._ZF_down_block(net, ksizes=[3,3], filters=[128,128],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,3), keep_prob = self.keep_prob)

            conv4, net = self._ZF_down_block(net, ksizes=[3,3], filters=[256,256],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,4), keep_prob = self.keep_prob)

            conv5, net = self._ZF_down_block(net, ksizes=[3,3], filters=[512,512],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,5), keep_prob = self.keep_prob)

            conv6, net = self._ZF_down_block(net, ksizes=[3,3], filters=[1024,1024],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/down%d"%(net_name,6), keep_prob = self.keep_prob, pool=False)

            net = self._ZF_up_block(net, conv5, ksizes=[3,3], filters=[512,512],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/up%d"%(net_name,5), keep_prob = self.keep_prob)

            net = self._ZF_up_block(net, conv4, ksizes=[3,3], filters=[256,256],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/up%d"%(net_name,4), keep_prob = self.keep_prob)

            net = self._ZF_up_block(net, conv3, ksizes=[3,3], filters=[128,128],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/up%d"%(net_name,3), keep_prob = self.keep_prob)

            net = self._ZF_up_block(net, conv2, ksizes=[3,3], filters=[64,64],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/up%d"%(net_name,2), keep_prob = self.keep_prob)

            net = self._ZF_up_block(net, conv1, ksizes=[3,3], filters=[32,32],
                activations=['relu']*2, strides=[1,1], batchnorm=[True]*2, dropout=dropout,
                name = "%s/up%d"%(net_name,1), keep_prob = self.keep_prob)

            net = self.conv_block(net, "%s/conv_block1"%(net_name), ksizes=[1], filters=[self.flags.classes],
                activations=['sigmoid'], strides=[1], batchnorm=[True])
 
            if self.flags.visualize and 'mask' in self.flags.visualize:
                tf.summary.image(name="masks", tensor=net,
                    max_outputs=3, collections=[tf.GraphKeys.IMAGES])

            self.logit = net

    def _ZF_down_block(self,net,ksizes,filters,dropout,keep_prob,name,activations,strides,batchnorm,pool=True):
        with tf.variable_scope(name.split('/')[-1]):
            net = self.conv_block(net, "%s/conv_block"%(name), ksizes=ksizes, filters=filters,
                activations=activations, strides=strides, batchnorm=batchnorm)

            if dropout:
                net = tf.nn.dropout(net, keep_prob = self.keep_prob)
            conv = net

            if pool:
                net = self._max_pool2D(net, ksize = [1,2,2,1], strides = [1,2,2,1],
                    padding = 'SAME', layer_name = '%s/pool'%(name))
        return conv, net

    def _ZF_up_block(self,net, down, ksizes,filters,dropout,keep_prob,name,activations,strides,batchnorm):
        channels = net.get_shape().as_list()[-1]
        with tf.variable_scope(name.split('/')[-1]):
            net = self._deconv2D(net, ksize=2, in_channel=channels, 
                out_channel=channels, strides=[1,2,2,1], layer_name="%s/deconv"%(name), 
                padding='SAME', activation=None, L2 = 1)

            try:
                net = tf.concat([net,down],axis=3)
            except:
                net = tf.concat(3, [net,down])

            net = self.conv_block(net, "%s/conv_block"%(name), ksizes=ksizes, filters=filters,
                activations=activations, strides=strides, batchnorm=batchnorm)

            if dropout:
                net = tf.nn.dropout(net, keep_prob = self.keep_prob)

        return net
