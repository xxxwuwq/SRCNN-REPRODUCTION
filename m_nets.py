import tensorflow as tf
from m_layers import *
import numpy as np


def srcnn(inputs, pad, name='srcnn'):

    # inputs_channels = inputs.get_shape().as_list()[-1]

    weight1 = m_weight_initializer([9, 9, 1, 64])
    conv_1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding=pad)
    # conv_1 = tf.cond(is_training, lambda: tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding='VALID'),
    #                  lambda: tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding='SAME'))
    relu_1 = tf.nn.relu(conv_1)

    weight2 = m_weight_initializer([1, 1, 64, 32])
    conv_2 = tf.nn.conv2d(input=relu_1, filter=weight2, strides=[1, 1, 1, 1], padding='SAME')
    relu_2 = tf.nn.relu(conv_2)

    weight3 = m_weight_initializer([5, 5, 32, 1])
    conv_3 = tf.nn.conv2d(input=relu_2, filter=weight3, strides=[1, 1, 1, 1], padding=pad)

    return conv_3

    net_structure = []
    HBF_KERNEL = tf.cast(tf.constant([[[[-1 / 12.]], [[2 / 12.]], [[-2 / 12.]], [[2 / 12.]], [[-1 / 12.]]],
                                      [[[2 / 12.]], [[-6 / 12.]], [[8 / 12.]], [[-6 / 12.]], [[2 / 12.]]],
                                      [[[-2 / 12.]], [[8 / 12.]], [[-12 / 12.]], [[8 / 12.]], [[-2 / 12.]]],
                                      [[[2 / 12.]], [[-6 / 12.]], [[8 / 12.]], [[-6 / 12.]], [[2 / 12.]]],
                                      [[[-1 / 12.]], [[2 / 12.]], [[-2 / 12.]], [[2 / 12.]], [[-1 / 12.]]]]), "float")
    # 512x512
    hbf_layer = tf.nn.conv2d(tf.cast(inputs, "float"), HBF_KERNEL, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
    net_structure.append(hbf_layer.get_shape().as_list())
    net = m_xu_group(hbf_layer, is_train, output_channels=8, conv_size=(5, 5), conv_stride=(1, 1), conv_padding='SAME',
                     activation=tf.nn.tanh, use_abs=True, pooling_size=(5, 5), pooling_stride=(2, 2),
                     pooling_padding='SAME',
                     name='conv1')
    net_structure.append(net.get_shape().as_list())
    net = m_xu_group(net, is_train, output_channels=16, conv_size=(5, 5), conv_stride=(1, 1), conv_padding='SAME',
                     activation=tf.nn.tanh, use_abs=False, pooling_size=(5, 5), pooling_stride=(2, 2),
                     pooling_padding='SAME',
                     name='conv2')
    net_structure.append(net.get_shape().as_list())
    net = m_xu_group(net, is_train, output_channels=32, conv_size=(1, 1), conv_stride=(1, 1), conv_padding='SAME',
                     activation=tf.nn.relu, use_abs=False, pooling_size=(5, 5), pooling_stride=(2, 2),
                     pooling_padding='SAME',
                     name='conv3')
    net_structure.append(net.get_shape().as_list())
    net = m_xu_group(net, is_train, output_channels=64, conv_size=(1, 1), conv_stride=(1, 1), conv_padding='SAME',
                     activation=tf.nn.relu, use_abs=False, pooling_size=(5, 5), pooling_stride=(2, 2),
                     pooling_padding='SAME',
                     name='conv4')
    net_structure.append(net.get_shape().as_list())
    net = m_xu_group(net, is_train, output_channels=128, conv_size=(1, 1), conv_stride=(1, 1), conv_padding='SAME',
                     activation=tf.nn.relu, use_abs=False, pooling_size=(16, 16), pooling_stride=(1, 1),
                     pooling_padding='VALID',
                     name='conv5')
    net_structure.append(net.get_shape().as_list())
    # 全连接层
    sp = net.get_shape().as_list()
    net = tf.reshape(net, [-1, sp[1] * sp[2] * sp[3]])
    # 权重设置
    weight_fc1 = m_weight_initializer([128, 2])
    bias_fc1 = m_bias_initializer([2])
    net = tf.nn.softmax(tf.matmul(net, weight_fc1) + bias_fc1)

    net_structure.append(net.get_shape().as_list())
    print(net_structure)
    return net






