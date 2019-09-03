import tensorflow as tf
from tensorflow.contrib import slim


class SRCNN:
    def __init__(self):
        pass

    def model(self, inputs, padding='VALID', name='srcnn'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            inputs = inputs / 255.0
            net = slim.conv2d(inputs, 64, [9, 9], padding=padding, scope='conv1_1')
            net = slim.conv2d(net, 32, [1, 1], padding=padding, scope='conv2_1')
            net = slim.conv2d(net, 1, [5, 5], padding=padding, activation_fn=None, scope='conv3_1')
        return net

    def __call__(self, inputs, padding='VALID', name='srcnn'):
        return self.model(inputs, padding, name)


