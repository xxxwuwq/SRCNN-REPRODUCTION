import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

# 权重初始化,0均值 0.01标准差
def m_weight_initializer(shape):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.01)
    return tf.Variable(initial)

# 偏置初始化
def m_bias_initializer(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层，无偏置
def m_conv2D_no_bias(inputs, output_channels, kernel_size=(3, 3), stride=(1, 1), padding='SAME', name='Conv2D'):
    inputs_channels = inputs.get_shape().as_list()[-1]
    weight = m_weight_initializer([*kernel_size, inputs_channels, output_channels])
    return tf.nn.conv2d(input=inputs, filter=weight, strides=[1, *stride, 1], padding=padding)

# 平均池化层
def m_avg_pooling(inputs, kernel_size=(3, 3), stride=(1, 1), padding='SAME'):
    return tf.nn.avg_pool(inputs, ksize=[1, *kernel_size, 1], strides=[1, *stride, 1], padding=padding)

# batch normalization
def m_batch_normalization(inputs, is_train, output_channels, epsilon=1e-4):
    beta = tf.Variable(tf.constant(0.0, shape=[output_channels]), name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[output_channels]), name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
    # m_train_pairs 0.3
    # sz_xu_net_pairs 0.01
    # m_wu_train_nets 0.1
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # previously 0.3

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    # 当is_train:True,mean_var_with_update
    # is_trainl:False,
    mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

# XuGuangshuo
def m_xu_group(inputs, is_train, output_channels, conv_size=(3, 3), conv_stride=(3, 3), conv_padding='SAME',
               use_abs=True, activation=tf.nn.relu, pooling_size=(3, 3), pooling_stride=(2, 2), pooling_padding='SAME',
               name='xu_group'):
    with tf.variable_scope(name):
        input_channels = inputs.get_shape().as_list()[-1]
        weight_conv = m_weight_initializer([*conv_size, input_channels, output_channels])
        output_channals = weight_conv.get_shape().as_list()[-1]
        conv_layer = m_conv2D_no_bias(inputs, output_channels, conv_size, conv_stride, conv_padding)
        if use_abs:
            conv_layer = tf.abs(conv_layer, name='absolute_layer')
        batch_norm_layer = m_batch_normalization(conv_layer, is_train, output_channals)
        activate_layer = activation(batch_norm_layer)
        pooling = m_avg_pooling(activate_layer, pooling_size, pooling_stride, padding=pooling_padding)
        return pooling

def gacc(inputs, sigama=0.5):
    return tf.exp(-tf.div(tf.square(inputs), tf.square(sigama)))





def bn(x, is_training, BN_DECAY):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)


    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-4)