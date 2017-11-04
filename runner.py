# encoding: utf-8

'''

@author: ZiqiLiu


@file: runner.py

@time: 2017/11/3 下午11:34

@desc:
'''
from reader import read_dataset
from models.CNN import CNN
import tensorflow as tf
from glob import glob
import os
import sys
import signal
from config import get_config


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.dataset = read_dataset(config.batch_size)
        self.graph = tf.Graph()
        self.model = None
        if not os.path.exists(self.config.model_path):
            os.mkdir(self.config.model_path)
        for key in config.__dict__:
            print(key, config.__dict__[key])
        with self.graph.as_default():
            self.model = CNN(self.config)
            tf.Graph.finalize(self.graph)

    def run(self):
        with self.graph.as_default(), tf.Session() as sess:

            model_path = os.path.join(self.config.model_path,
                                      self.config.model_name)
            saver = tf.train.Saver()
            files = glob(os.path.join(self.config.model_path, '*.ckpt.*'))

            if len(files) > 0:
                saver.restore(sess, model_path)
                print(('Model restored from:' + model_path))
            else:
                print("Model doesn't exist.\nInitializing........")
                sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def handler_stop_signals(signum, frame):

                print(
                    'training shut down,  the model will be save in %s' % (
                        model_path))
                saver.save(sess, save_path=model_path)
                sys.exit(0)

            signal.signal(signal.SIGINT, handler_stop_signals)
            signal.signal(signal.SIGTERM, handler_stop_signals)

            while self.dataset.epoch < self.config.max_epoch:
                data, label = self.dataset.next_training_batch()
                _, step, loss = sess.run(
                    [self.model.train_op, self.model.global_step,
                     self.model.loss],
                    feed_dict={self.model.input: data,
                               self.model.label: label})
                if step % 50 == 0:
                    print('step %d, epoch %d, loss %f' % (
                        step, self.dataset.epoch, loss))
                if (step) % self.config.valid_step == 0:
                    valid_data, valid_label = self.dataset.valid_batch()
                    accu = sess.run(self.model.accuracy,
                                    feed_dict={self.model.input: valid_data,
                                               self.model.label: valid_label})
                    print('valid accuracy: %f' % accu)

            saver.save(sess, save_path=model_path)
            print(
                'training finished,  the model will be save in %s' % (
                    self.config.model_path))
            self.test()

    def test(self):
        with self.graph.as_default(), tf.Session() as sess:
            files = glob(os.path.join(self.config.model_path, '*.ckpt.*'))
            assert len(files) > 0

            self.model = CNN(self.config)
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.config.model_path,
                                             self.config.model_name))
            print(('Model restored from:' + self.config.model_path))


if __name__ == '__main__':
    runner = Runner(get_config())
    runner.run()
