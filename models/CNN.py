# encoding: utf-8

'''

@author: ZiqiLiu


@file: CNN.py

@time: 2017/11/3 下午10:39

@desc:
'''
import tensorflow as tf


class CNN(object):
    def __init__(self, config):
        self.config = config
        if config.initializer == 'xavier':
            self.initializer = tf.contrib.layers.xavier_initializer_conv2d()
        else:
            self.initializer = tf.truncated_normal_initializer(stddev=0.1)

        if config.activate_func == 'sigmoid':
            self.activate_func = tf.nn.sigmoid
        elif config.activate_func == 'relu':
            self.activate_func = tf.nn.relu
        elif config.activate_func == 'tanh':
            self.activate_func = tf.nn.tanh
        else:
            raise Exception('activation function not defined!')

        self.input = tf.placeholder(tf.float32, [None, 28, 28], name='input')
        self._input = tf.expand_dims(self.input, 3)
        self.label = tf.placeholder(tf.float32, [None, 10], name='label')

        # first conv+pooling
        self.conv1_w = tf.get_variable("conv1_w", [5, 5, 1, 32], tf.float32,
                                       initializer=self.initializer)
        self.conv1_b = tf.get_variable("conv1_b", [32], tf.float32,
                                       initializer=tf.zeros_initializer
                                       )
        tf.layers.conv2d()

        self.h1_conv = self.activate_func(
            self.conv2d(self._input, self.conv1_w) + self.conv1_b, 'hidden1')
        self.h1 = self.max_pool_2x2(self.h1_conv, 'pooling1')

        # second conv+pooling
        self.conv2_w = tf.get_variable("conv2_w", [5, 5, 32, 64], tf.float32,
                                       initializer=self.initializer)
        self.conv2_b = tf.get_variable("conv2_b", [64], tf.float32,
                                       tf.zeros_initializer)

        self.h2_conv = self.activate_func(
            self.conv2d(self.h1, self.conv2_w) + self.conv2_b, 'hidden2')
        self.h2 = self.max_pool_2x2(self.h2_conv, 'pooling2')

        self.flatten = tf.reshape(self.h2, [-1, 7 * 7 * 64], 'flatten')

        # fc1
        self.fc1_w = tf.get_variable('fc1_w', [7 * 7 * 64, 1024], tf.float32,
                                     self.initializer)
        self.fc1_b = tf.get_variable('fc1_b', [1024], tf.float32,
                                     tf.zeros_initializer)
        self.fc1 = self.activate_func(tf.matmul(self.flatten, self.fc1_w) + self.fc1_b,
                              'fc1')

        # dropout
        if config.dropout:
            self.dropout = tf.nn.dropout(self.fc1, config.keep_prob)
        else:
            self.dropout = self.fc1

        # fc2
        self.fc2_w = tf.get_variable('fc2_w', [1024, 10], tf.float32,
                                     self.initializer)
        self.fc2_b = tf.get_variable('fc2_b', [10], tf.float32,
                                     tf.zeros_initializer)
        self.fc2 = self.activate_func(tf.matmul(self.fc1, self.fc2_w) + self.fc2_b,
                              'fc2')
        self.softmax = tf.nn.softmax(logits=self.fc2, name='softmax')

        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(self.softmax, 1), tf.argmax(self.label, 1)),
            tf.float32), name='accuracy')

        # loss and gradient
        self.global_step = tf.Variable(0, trainable=False)
        initial_learning_rate = tf.Variable(
            config.learning_rate, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            initial_learning_rate, self.global_step, self.config.decay_step,
            self.config.lr_decay,
            name='lr') if config.use_lr_decay else initial_learning_rate
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.loss = -tf.reduce_sum(self.label * tf.log(self.softmax))
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)
