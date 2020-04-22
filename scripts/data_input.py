"""
Data input preparation from decoding TFrecords, onehot encoding, augmentation, and batching 2.0

Created on 03/19/2019

@author: RH
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys, cv2
import numpy as np

class DataSet(object):
    # bs is batch size; ep is epoch; images are images; mode is test/train; filename is tfrecords
    def __init__(self, bs, cls=2, images=None, mode=None, filename=None):
        self._batchsize = bs
        self._index_in_epoch = 0
        self._images = images
        self._mode = mode
        self._filename = filename
        self._classes = cls

    # decoding tfrecords for real test
    def Real_decode(self, serialized_example):
        features = tf.parse_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={'test/image': tf.FixedLenFeature([], tf.string)})

        image = tf.decode_raw(features['test/image'], tf.float32)
        image = tf.reshape(image, [-1, 299, 299, 3])

        return image

    # dataset preparation; batching; Real test or not; train or test
    def data(self):
        batch_size = self._batchsize
        filenames = tf.placeholder(tf.string, shape=None)
        dataset = tf.data.TFRecordDataset(filenames)
        batched_dataset = dataset.batch(batch_size, drop_remainder=False)
        batched_dataset = batched_dataset.map(self.Real_decode)
        iterator = batched_dataset.make_initializable_iterator()
        return iterator, self._filename, filenames


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# used for tfrecord images generation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    imlist = []
    for a in os.listdir('../top_100_tiles'):
        if 'jpeg' in a:
            imlist.append(str('../top_100_tiles/'+a))

    filename = '../Results/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(imlist)):
        if not i % 1000:
            sys.stdout.flush()
        try:
            # Load the image
            img = load_image(imlist[i])
            # Create a feature
            feature = {'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image:' + imlist[i])
            pass

    writer.close()
    sys.stdout.flush()
