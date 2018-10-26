import tensorflow as tf
import os
from os.path import join
import random


class Dataset:
    def __init__(self, dir):
        self.dir = dir

        # Get all file names in directory
        benign_files = [join(dir, 'benign', f) for f in os.listdir(join(self.dir, 'benign'))]
        malignant_files = [join(dir, 'malignant', f) for f in os.listdir(join(self.dir, 'malignant'))]
        fnames = benign_files + malignant_files
        self.labels = [0] * len(benign_files) + [1] * len(malignant_files)

        # Shuffle the data
        combined = list(zip(fnames, self.labels))
        random.shuffle(combined)
        fnames, self.labels = zip(*combined)

        fnames_t = tf.constant(fnames)
        labels_t = tf.constant(self.labels)

        images = tf.data.Dataset.from_tensor_slices(fnames_t).map(self._parse_function, num_parallel_calls=4)
        labels = tf.data.Dataset.from_tensor_slices(labels_t).map(lambda l: tf.one_hot(l, 2), num_parallel_calls=4)
        self.dataset = tf.data.Dataset.zip((images, labels))

        self.size = len(benign_files) + len(malignant_files)

    @staticmethod
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        return image_decoded
