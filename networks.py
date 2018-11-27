import tensorflow as tf
from tensorflow.contrib import slim


def srcnn(inputs, padding='VALID', name='srcnn'):
    with tf.variable_scope(name):
        net = slim.conv2d(inputs, 64, [9, 9], padding=padding, scope='conv1_1')
        net = slim.conv2d(net, 32, [1, 1], padding=padding, scope='conv2_1')
        net = slim.conv2d(net, 1, [5, 5], padding=padding, scope='conv3_1')
    return net
