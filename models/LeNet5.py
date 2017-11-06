# encoding: utf-8

'''

@author: ZiqiLiu


@file: LeNet5.py

@time: 2017/11/3 下午10:39

@desc:
'''
import tensorflow as tf


class LeNet5(object):
    def __init__(self, config):
        self.config = config
        # collect layers to calculate MI
        self.layers_collector = []

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
        self.h1_conv = self.conv2d(self._input, 32, [5, 5], name='hidden1')
        self.h1 = tf.layers.max_pooling2d(self.h1_conv, [2, 2], [2, 2],
                                          name='pooling1')

        # second conv+pooling
        self.h2_conv = self.conv2d(self.h1, 64, [5, 5], name='hidden2')
        self.h2 = tf.layers.max_pooling2d(self.h2_conv, [2, 2], [2, 2],
                                          name='pooling2')

        # flatten
        self.flatten = tf.reshape(self.h2, [-1, 7 * 7 * 64], 'flatten')

        # fc1
        self.fc1 = tf.layers.dense(self.flatten, 1024, self.activate_func,
                                   kernel_initializer=self.initializer,
                                   name='fc1')

        # dropout
        if config.dropout:
            self.dropout = tf.nn.dropout(self.fc1, config.keep_prob)
        else:
            self.dropout = self.fc1

        self.fc2 = tf.layers.dense(self.fc1, 10,
                                   kernel_initializer=self.initializer,
                                   name='fc2')
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
        if self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate)

        self.loss = -tf.reduce_sum(self.label * tf.log(self.softmax))
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=self.global_step)

        self.layers_collector.append(self.input)
        self.layers_collector.append(self.transpose(self.h1))
        self.layers_collector.append(self.transpose(self.h2))
        self.layers_collector.append(tf.expand_dims(self.fc1, 1))
        self.layers_collector.append(self.softmax)

    def transpose(self, layer):
        return tf.transpose(layer, [0, 3, 1, 2])

    def conv2d(self, input, channel, kernel, name=None):
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=self.config.l2_beta) if self.config.l2_norm else None

        conv = tf.layers.conv2d(input, channel, kernel,
                                strides=(1, 1), padding='SAME',
                                use_bias=True,
                                kernel_initializer=self.initializer,
                                kernel_regularizer=l2_regularizer)
        if self.config.batch_norm:
            conv = tf.layers.batch_normalization(conv)
        activate = self.activate_func(conv, name)
        return activate
