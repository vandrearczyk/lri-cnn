"""
S-LRI implementation for the synthetic texture dataset.
"""

import tensorflow as tf
import sys
sys.path.insert(0, '../')
from sh_networks_utils import *
from utils import *


IM_SIZE = 32
CHANNELS = 1

def SHconv3d(name, l_input, ksize, b, out_channels, degreeMax=1, M=4, stride=1, is_trainable=True, is_hn=False):
    '''
    Returns the s-lri response (pooled on the orientation channels)
    '''
    stddev = 1.  
    return tf.nn.bias_add(s_conv3d(l_input, out_channels, ksize, [1,  stride, stride, stride, 1], 'VALID', degreeMax, stddev, name, M, is_trainable, is_hn),b)

def build_model(X, batch_size, n_class, nf1, ksize, stride=1, is_trainable=True, degreeMax=1, is_shconv=True, is_hn=False, M=24):
    '''
    Builds the model and returns the output for training and infering
    X: input volumes
    batch_size: batch size
    n_class: number of classes
    nf1: number of filters
    ksize: kernel size
    stride: convolution stride
    is_trainable: bool to evaluate without training the lri layer
    degreeMax: maximum degree of the SHs
    is_hn: whether we have one radial profile per degree n (non-polar-separable) 
    M: number of orientations sampled for the local rotation invariance
    '''
    if is_shconv:
        bc1 = tf.get_variable('b_c1', shape=nf1, initializer=tf.constant_initializer(0.),trainable=is_trainable)
        conv1 = SHconv3d('conv1', X, ksize, bc1, nf1, degreeMax, M, stride, is_trainable,is_hn)
        print('conv1.shape: ',conv1.shape)
    else: # Normal conv layer
        bc1 = tf.get_variable('b_c1', shape=nf1, initializer=tf.constant_initializer(1e-2),trainable=is_trainable)
        wc1 = tf.get_variable('w_c1', shape=[ksize,ksize,ksize,1,nf1],initializer=tf.contrib.layers.xavier_initializer(),trainable= is_trainable)
        conv1 = conv3d('conv1', X, wc1, bc1, stride)
        print('conv1.shape: ',conv1.shape)
    conv1 = tf.nn.relu(conv1,'relu1')
    
    # global average pool
    gap = gavg_pool('gap', conv1)
    print('gavgpool shape', gap.shape)
    
    # Output: class prediction
    bout = tf.get_variable('b_out', shape=n_class, initializer=tf.constant_initializer(1e-2),trainable=True)
    wout = tf.get_variable('w_out', shape=[nf1,n_class], initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    out = tf.matmul(gap, wout) + bout
    return out
