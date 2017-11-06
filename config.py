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
        self.learning_rate = 5e-4
        self.max_epoch = 40
        self.valid_step = 600
        self.initializer = 'xavier'  # xavier or normal
        self.activate_func = 'tanh'  # sigmoid or relu or tanh

        self.dropout = True
        self.keep_prob = 0.6

        self.batch_norm = False

        self.l2_norm = False
        self.l2_beta = 0.01

        self.use_lr_decay = True
        self.lr_decay = 0.85
        self.decay_step = 1200

        # sample size for calculating information entropy
        self.sample_size = 2000
        self.info_plane_interval = 1200  # in step

        self.model_path = './trained_model/'
        self.model_name = 'latest.ckpt'
