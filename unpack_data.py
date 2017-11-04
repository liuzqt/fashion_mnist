# encoding: utf-8

'''

@author: ZiqiLiu


@file: unpack_data.py

@time: 2017/11/3 下午6:38

@desc:
'''

import numpy as np
import struct

train_images_idx3_ubyte_file = './data/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = './data/train-labels-idx1-ubyte'

test_images_idx3_ubyte_file = './data/t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = './data/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        bin_data = f.read()

    # parse header
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    print('total images: %d, image size: %d*%d' % (
        num_images, num_rows, num_cols))

    # parse data
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('parsed %d' % (i + 1))
        images[i] = np.array(
            struct.unpack_from(fmt_image, bin_data, offset)).reshape(
            (num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        bin_data = f.read()

    # parse header
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('labels number: %d' % (num_images))

    # parse data
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print(
                'parsed %d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels.astype(np.uint8)


def run():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)

    np.save('./data/train_data.npy', train_images)
    np.save('./data/train_label.npy', train_labels)
    np.save('./data/test_data.npy', test_images)
    np.save('./data/test_label.npy', test_labels)


if __name__ == '__main__':
    run()
