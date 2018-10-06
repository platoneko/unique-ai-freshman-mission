# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import time

import inference
import train


time_delay = 10


def evaluate(x_test, y_test):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [100, 28, 28, 1], name='x-test')
        y = tf.placeholder(tf.int32, [100], name='y-test')

        y_hat = inference.inference(x, regularizer=None, training=False)

        correct = tf.equal(y, tf.cast(tf.argmax(y_hat, 1), tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        var_avg = tf.train.ExponentialMovingAverage(train.moving_avg_decay)
        var_to_restore = var_avg.variables_to_restore()
        saver = tf.train.Saver(var_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
                    cur_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('After {} step(s), acc = {}'.format(cur_step, acc))
                    if int(cur_step) >= train.max_step:
                        return
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(time_delay)


def main(argv=None):
    mnist = np.load('mnist.npz')
    x_test = mnist['x_test'][:100].reshape(100, 28, 28, 1)
    y_test = mnist['y_test'][:100]
    evaluate(x_test, y_test)


if __name__ == '__main__':
    tf.app.run()