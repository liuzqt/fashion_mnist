# encoding: utf-8

'''

@author: ZiqiLiu


@file: random_forest.py

@time: 2017/11/3 下午6:34

@desc: try random forest as a quick base line
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

train_data = np.load('./data/train_data.npy')
train_data = np.reshape(train_data, [train_data.shape[0], -1])
train_label = np.load('./data/train_label.npy')
test_data = np.load('./data/test_data.npy')
test_data = np.reshape(test_data, [test_data.shape[0], -1])
test_label = np.load('./data/test_label.npy')

print('random forest baseline')
rf = RandomForestClassifier(n_estimators=70, n_jobs=-1, bootstrap=True)
rf.fit(train_data, train_label)
print('training finished')
accuracy = rf.score(test_data, test_label)
print('test accuracy: %f' % accuracy)
