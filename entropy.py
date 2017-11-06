#!/usr/bin/env python3
# encoding: utf-8

'''

@author: ZiqiLiu


@file: entropy.py

@time: 2017/11/3 下午3:31

@desc:
'''

import numpy as np

img_bins = np.arange(0, 256)
hidden_bins = np.arange(1, 31)
digitize_bins = np.linspace(-1, 1, 30)


def entropy(layers):
    """
    
    :param layers: [layers,batch,....] first is input, last is output
    :return: IXY, ITY of shape [layers]
    """
    ent = [[] for i in range(len(layers) - 2)]
    # input
    input_ent = np.asarray([_get_entropy(img, img_bins) for img in layers[0]])
    output_ent = np.asarray([_entropy(softmax) for softmax in layers[-1]])

    for i, layer in enumerate(layers[1:-1]):
        for sample in layer:
            channel_ent = [
                _get_entropy(np.digitize(channel, digitize_bins), hidden_bins)
                for
                channel in
                sample]
            avg_ent = np.asarray(channel_ent).mean()
            ent[i].append(avg_ent)
    ent = np.asarray(ent)
    IXT = []
    ITY = []
    for layer_ent in ent:
        IXT.append(layer_ent - input_ent)
        ITY.append(output_ent - layer_ent)
    IXT = np.asarray(IXT)
    ITY = np.asarray(ITY)
    IXT = np.mean(IXT, 1)
    ITY = np.mean(ITY, 1)
    return IXT, ITY


def _get_entropy(band, bins):
    hist, _ = np.histogram(band, bins=bins)
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    return _entropy(prob)


def _entropy(probs):
    return -(np.log2(probs) * probs).sum()

# im = imread('ex.jpg',as_grey=True)
