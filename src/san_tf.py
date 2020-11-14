# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:12:17 2020

"""

import yaml


import tensorflow as tf
import tensorflow.keras as kl
import tensorflow.keras.layers as lys

class SAM(lys.Layer):
    
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.dilation = dilation
        self.rel_planes = rel_planes
        self.out_planes = out_planes
        
        self.conv1 = conv1x1(rel_planes)
        self.conv2 = conv1x1(rel_planes)
        self.conv3 = conv1x1(out_planes)
        if sa_type == 0:
            raise NotImplementedError('pairwise self-attention is not implemented')

        else:
            self.conv_w = kl.Sequential([lys.BatchNormalization(), lys.ReLU(),
                                        lys.Conv2D(out_planes // share_planes, 1, strides=1,padding='same',use_bias=False),
                                        lys.BatchNormalization(), lys.ReLU(),
                                        lys.Conv2D(pow(kernel_size, 2) * out_planes // share_planes, 1, strides=1,padding='same')])

        
    def call(self, x):
        #x3 is Beta(xj)
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            raise NotImplementedError('pairwise self-attention is not implemented')
        else:  # patchwise
            if self.stride != 1:
                x1 = tf.image.extract_patches(x, (1,1,1,1), 
                                              (1,self.stride,self.stride,1),
                                              (1,self.dilation,self.dilation,1),
                                              'VALID')
            x1 = tf.reshape(x1, (tf.shape(x)[0], 1, x.shape[1]*x.shape[2], self.rel_planes)) #TODO:check if this shape is valid or not
            x2 = tf.reshape(tf.image.extract_patches(x2, (1,self.kernel_size,self.kernel_size,1),
                                          (1,self.stride,self.stride,1), 
                                          (1,self.dilation,self.dilation,1),
                                          'SAME'),(tf.shape(x)[0], 
                                                   1, 
                                                   x.shape[1]*x.shape[2], 
                                                   pow(self.kernel_size, 2)*self.rel_planes))
                                                   
            w = tf.reshape(self.conv_w(tf.concat([x1, x2],axis=-1)),(tf.shape(x)[0],
                                                                 pow(self.kernel_size, 2),
                                                                 x.shape[1]*x.shape[2], 
                                                                 -1))
           
            #aggregation
            x3 = tf.reshape(tf.image.extract_patches(x3, (1,self.kernel_size,self.kernel_size,1),
                                          (1,self.stride,self.stride,1), 
                                          (1,self.dilation,self.dilation,1),
                                          'SAME'),(tf.shape(x)[0], pow(self.kernel_size, 2), x.shape[1]*x.shape[2], -1))
            
        ret = tf.reduce_sum(tf.math.multiply(x3,
                                           tf.repeat(w,tf.shape(x3)[-1]//tf.shape(w)[-1],axis=-1),
                                           name='sam_mal'),
                          axis=1,name='sam_reduce_sm', keepdims=True)
        return tf.reshape(ret,(tf.shape(x)[0], x.shape[1], x.shape[2], self.out_planes))

        

class Bottleneck(lys.Layer):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = lys.BatchNormalization()
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = lys.BatchNormalization()
        self.conv = conv1x1(out_planes)
        # Although below code looks conv1x1, it isn't use conv1x1
        # self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = lys.ReLU()

    def call(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

def conv1x1(out_ch):
    return lys.Conv2D(out_ch, 1, strides=1,padding='same')

class SAN():
    def __init__(self, sa_type, block, layers, kernels, num_classes):
        c = 64                
        self.layer_in = kl.Sequential([conv1x1(c), 
                                        lys.BatchNormalization(),
                                        lys.ReLU()])
        
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.avgpool = lys.GlobalAveragePooling2D()
        self.flatten = lys.Flatten()
        self.fc = lys.Dense(num_classes, name='out_dense')
        
    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        layers.append(lys.MaxPool2D(pool_size=2, strides=2))
        layers.append(conv1x1(planes))
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        layers.append(lys.BatchNormalization())
        layers.append(lys.ReLU())
        return kl.Sequential(layers)

    def __call__(self, x):        
        
        out = self.layer_in(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return kl.Model(inputs=x, outputs=out)


def san(input_shape, sa_type, layers, kernels, num_classes):
    net = SAN(sa_type, Bottleneck, layers, kernels, num_classes)
    Input = kl.Input(input_shape)
    return net(Input)
