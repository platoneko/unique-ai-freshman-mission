# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import os

import inference


regularization_rate = 0.0001
moving_avg_decay = 0.99
base_learning_rate = 0.0001
exp_decay_rate = 0.99
batch_size = 100
max_step = 3000
model_path = 'model/'
model_name = 'model.ckpt'


def train(x_train, y_train):
    x = tf.placeholder(tf.float32, [batch_size, inference.image_size, inference.image_size, inference.channel_num],
                       name='x-train')
    y = tf.placeholder(tf.int32, [batch_size], name='y-train')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    y_hat = inference.inference(x, regularizer=regularizer, training=True)

    global_step = tf.Variable(0, trainable=False)

    var_avg = tf.train.ExponentialMovingAverage(moving_avg_decay)
    var_avg_op = var_avg.apply(tf.trainable_variables())

    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))

    learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, x_train.shape[0]/batch_size,
                                               exp_decay_rate)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_op = tf.group(train_step, var_avg_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epoch_step = x_train.shape[0]//batch_size
        for i in range(max_step):
            if i % (epoch_step) == 0 and i >= epoch_step:
                saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)
                print("epoch = {}, loss = {}".format(cur_step//epoch_step, loss_value))
            head = (i * batch_size) % x_train.shape[0]
            tail = head + batch_size
            _, loss_value, cur_step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: x_train[head:tail], y: y_train[head:tail]})

        saver.save(sess, os.path.join(model_path, model_name), global_step=global_step)
        print("epoch = {}, loss = {}".format(cur_step // epoch_step, loss_value))


def main(argv=None):
    mnist = np.load('mnist.npz')
    x_train = mnist['x_train'].reshape(60000, 28, 28, 1)
    y_train = mnist['y_train']
    train(x_train, y_train)


if __name__ == '__main__':
    tf.app.run()





