import tensorflow as tf
import os
from os.path import join


class Dataset():
    def __init__(self, h, w, dir):
        self.h = h
        self.w = w
        self.dir = dir
        self.size = len(os.listdir(dir))

    def _parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.h, self.w])
        return image_resized

    def iterator(self):
        fnames = [join(self.dir, f) for f in os.listdir(self.dir)]
        fnames_t = tf.constant(fnames)
        labels_t = tf.constant([1] * len(fnames))

        images = tf.data.Dataset.from_tensor_slices(fnames_t).map(self._parse_function, num_parallel_calls=4)
        labels = tf.data.Dataset.from_tensor_slices(labels_t).map(lambda l: tf.one_hot(l, 2), num_parallel_calls=4)
        dataset = tf.data.Dataset.zip((images, labels)).shuffle(1000).batch(32).repeat().prefetch(32)
        return dataset.make_one_shot_iterator()