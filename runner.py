# encoding: utf-8

'''

@author: ZiqiLiu


@file: runner.py

@time: 2017/11/3 下午11:34

@desc:
'''
from reader import read_dataset
from models.LeNet5 import LeNet5
from models.ResNet import ResNet
import tensorflow as tf
from glob import glob
import os
import sys
import signal
from config import get_config
from entropy import entropy
from plot import plot_info_plain
import pickle



class Runner(object):
    def __init__(self, config, model):
        self.config = config
        self.dataset = read_dataset(config.batch_size, config.valid_size,
                                    config.sample_size)
        self.graph = tf.Graph()
        self.model = None
        self.restore = False
        if not os.path.exists(self.config.model_path):
            os.mkdir(self.config.model_path)
        for key in config.__dict__:
            print(key, config.__dict__[key])
        with self.graph.as_default():
            self.model = model(self.config)

        self.IXT = []
        self.ITY = []

    def run(self):
        with self.graph.as_default(), tf.Session() as sess:
            self.restore = True
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
                _, step, loss, layers = sess.run(
                    [self.model.train_op, self.model.global_step,
                     self.model.loss, self.model.layers_collector],
                    feed_dict={self.model.input: data,
                               self.model.label: label})

                if step % self.config.valid_step == 0:
                    valid_data, valid_label = self.dataset.valid_batch()
                    accu = sess.run(self.model.accuracy,
                                    feed_dict={self.model.input: valid_data,
                                               self.model.label: valid_label})
                    print('step %d, epoch %d, valid accuracy: %f' % (
                        step, self.dataset.epoch, accu))
                if step % self.config.info_plane_interval == 0:
                    print('flag')
                    sample_data = self.dataset.sample_batch()
                    layers = sess.run(self.model.layers_collector,
                                      feed_dict={self.model.input: sample_data
                                                 })
                    ixt, ity = entropy(layers)

                    self.IXT.append(ixt)
                    self.ITY.append(ity)

            self._test(sess)
            saver.save(sess, save_path=model_path)
            print(
                'training finished,  the model will be save in %s' % (
                    self.config.model_path))

    def test(self):
        with self.graph.as_default(), tf.Session() as sess:
            files = glob(os.path.join(self.config.model_path, '*.ckpt.*'))
            assert len(files) > 0
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.config.model_path,
                                             self.config.model_name))
            print(('Model restored from:' + self.config.model_path))
            self._test(sess)
            self.plot_info_plane()

    def plot_info_plane(self):
        with open('ixt', 'wb') as f:
            pickle.dump(self.IXT, f)
        with open('ity', 'wb') as f:
            pickle.dump(self.ITY, f)
        plot_info_plain(self.IXT, self.ITY)

    def _test(self, sess):
        test_data, test_label = self.dataset.test_batch()
        accu = sess.run(self.model.accuracy,
                        feed_dict={self.model.input: test_data,
                                   self.model.label: test_label})
        print('test accuracy:%f' % accu)


if __name__ == '__main__':
    runner = Runner(get_config(), LeNet5)
    runner.run()
    # runner.test()
