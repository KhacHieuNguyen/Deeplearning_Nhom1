import tensorflow as tf
import math 
import pdb
from hparams import hparams
import cv2
import numpy as np

class Dataset(object):
    def __init__(self, hparams, record_path):
        self.hparams = hparams
        self.record_path = record_path
        zero = tf.zeros([1], dtype=tf.int64)
        self.keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature([], tf.string, default_value='jpg'),
            'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/height': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            # 'image/class':tf.io.FixedLenFeature([hparams.num_point*2], tf.float32),
            'image/mask': tf.io.FixedLenFeature([], tf.string, default_value='')
        }

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.cast(tf.io.decode_jpeg(res['image/encoded'], 3), tf.float32)/255.0
        
        image = tf.image.resize(image, (hparams.max_height , hparams.max_width))
        mask  =  tf.cast(tf.io.decode_jpeg(res['image/mask'] ,1) > 0, tf.uint8)
        # mask = tf.image.resize(mask, (hparams.max_height , hparams.max_width))
        #pdb.set_trace()
        # label = res['image/class']
        # #label = label[:4] * hparams.max_width
        # #label = label[4:] * hparams.max_height
        # #a=1
        print(image.shape,mask.shape)

        # cv2.imwrite( "result.jpg", tf.make)
        return image, mask

    def load_tfrecord(self, repeat=None):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        self.dataset = dataset.batch(self.hparams.batch_size)
        #self.dataset = dataset.cache()
        self.iterator = iter(dataset)

    def next_batch(self):
        return self.iterator.get_next()
if __name__ == "__main__":
    dataset = Dataset(hparams,"data/data_gen.train")
    dataset.load_tfrecord()
    print(dataset.dataset)
    