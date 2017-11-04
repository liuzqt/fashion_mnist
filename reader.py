# encoding: utf-8

'''

@author: ZiqiLiu


@file: reader.py

@time: 2017/11/3 下午9:21

@desc:
'''
import numpy as np


def one_hot(label, num_classes):
    '''
    
    :param label: ndarray of shape (n)
    :param num_classes: 
    :return: ndarray of shape (n,num_classes) 
    '''
    oh = np.zeros((label.shape[0], num_classes))
    oh[np.arange(label.shape[0]), label] = 1
    return oh


class DataSet(object):
    def __init__(self, batch_size, valid_size):
        self.batch_size = batch_size
        self.num_classes = 10
        self.valid_size = valid_size

        self.train_data = np.load('./data/train_data.npy')
        self.train_label = one_hot(np.load('./data/train_label.npy'),
                                   num_classes=self.num_classes)
        assert len(self.train_data) == len(self.train_label)

        # I split a small part from train data as validation
        self.valid_data, self.train_data = np.split(self.train_data,
                                                    [self.valid_size, ])
        self.valid_label, self.train_label = np.split(self.train_label,
                                                      [self.valid_size, ])

        self.test_data = np.load('./data/test_data.npy')
        self.test_label = one_hot(np.load('./data/test_label.npy'),
                                  num_classes=self.num_classes)
        assert len(self.test_data) == len(self.test_label)

        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

        # for convenience, only use batch_size as factor or data size
        assert len(self.train_data) % self.batch_size == 0

        self._index_range = np.arange(0, len(self.train_data))
        self.shuffle_index = np.copy(self._index_range)
        np.random.shuffle(self.shuffle_index)

        self.epoch = 0
        self.pos = 0

    def next_training_batch(self):
        data = self.train_data[
            self.shuffle_index[self.pos:self.pos + self.batch_size]]
        label = self.train_label[
            self.shuffle_index[self.pos:self.pos + self.batch_size]]
        self.pos += self.batch_size
        if self.pos == self.train_size:
            self.epoch += 1
            np.random.shuffle(self.shuffle_index)
        return data, label

    def valid_batch(self):
        return self.valid_data, self.valid_label

    def test_batch(self):
        for i in range(0, self.test_size, self.batch_size):
            data = self.test_data[i:i + self.batch_size]
            label = self.test_label[i:i + self.batch_size]
            yield data, label


def read_dataset(batch_size=50, valid_size=1000):
    return DataSet(batch_size, valid_size)
