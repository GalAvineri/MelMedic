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
        self.fnames = benign_files + malignant_files
        self.labels = [0] * len(benign_files) + [1] * len(malignant_files)

        # Shuffle the data
        combined = list(zip(self.fnames, self.labels))
        random.shuffle(combined)
        self.fnames, self.labels = zip(*combined)

        fnames_t = tf.constant(self.fnames)
        labels_t = tf.constant(self.labels)

        images = tf.data.Dataset.from_tensor_slices(fnames_t)
        labels = tf.data.Dataset.from_tensor_slices(labels_t)
        self.dataset = tf.data.Dataset.zip((images, labels))

        self.size = len(self.labels)

    @staticmethod
    def parse_sample(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        label = tf.one_hot(label, 2)
        return image_decoded, label
