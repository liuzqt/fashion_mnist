# encoding: utf-8

'''

@author: ZiqiLiu


@file: config.py

@time: 2017/11/3 下午11:41

@desc:
'''


def get_config():
    return Config()


class Config(object):
    def __init__(self):
        self.batch_size = 50
        self.valid_size = 1000
        self.learning_rate = 1e-3
        self.max_epoch = 10
        self.valid_step = 600
        self.initializer = 'normal'
        self.activate_func = 'tanh' # sigmoid or relu

        self.dropout = True
        self.keep_prob = 0.6

        self.use_lr_decay = True
        self.lr_decay = 0.8
        self.decay_step = 3000

        self.model_path = './trained_model/'
        self.model_name = 'latest.ckpt'
