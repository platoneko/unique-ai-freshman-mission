# -*- coding: utf-8 -*-


import tensorflow as tf


input_node = 784
output_node = 10

image_size = 28
channel_num = 1

conv1_size = 5
conv1_deep = 32

conv2_size = 5
conv2_deep = 64

fc1_node = 512


def get_weight_variable(shape, regularizer):
    weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('loss', regularizer(weight))
    return weight


def inference(x, regularizer=None, training=True):
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight', [conv1_size, conv1_size, channel_num, conv1_deep],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias', [conv1_deep], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weigth', [conv2_size, conv2_size, conv1_deep, conv2_deep],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable('bias', [conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool2_shape = pool2.get_shape().as_list()
    pool2_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    pool2_vector = tf.reshape(pool2, [pool2_shape[0], pool2_size])

    with tf.variable_scope('layer5-fc1'):
        fc1_weight = get_weight_variable([pool2_size, fc1_node], regularizer)
        fc1_bias = tf.get_variable('bias', [fc1_node], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weight) + fc1_bias)
        if training:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weight = get_weight_variable([fc1_node, output_node], regularizer)
        fc2_bias = tf.get_variable('bias', [output_node], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weight) + fc2_bias

    return fc2

