from auxiliries import io

import os
from os.path import join
import json
import shutil
import argparse
import multiprocessing


def filter_invalid_images(imgs_dir, dscs_dir, inv_imgs_dir, inv_dscs_dir, num_processes):
    """
    Moves all invalid descriptions and their corresponding images files
    out of the images and description directories and into directories marked as invalid elements

    A valid descriptions has:
    1. meta -> clinical -> benign_malignant key path
    2. a label which is either 'benign' or 'malignant'. e.g  not 'None' or 'indeterminate'.
    :param imgs_dir:
    :param dscs_dir:
    :param inv_imgs_dir:
    :param inv_dscs_dir:
    :return:
    """
    """
    Find whether the 
    """
    io.validate_exists_and_dir(imgs_dir, 'imgs_dir')
    io.validate_exists_and_dir(dscs_dir, 'dscs_dir')
    # Create the result dirs
    io.create_or_recreate_dir(inv_imgs_dir)
    io.create_or_recreate_dir(inv_dscs_dir)

    # Find all the descriptions
    dscs_fnames = os.listdir(dscs_dir)

    # Find which descriptions are invalid
    # And mark them and their corresponding image for moving
    src_paths = []
    dst_paths = []
    for dsc_fname in dscs_fnames:
        valid_dsc = True
        dsc_path = join(dscs_dir, dsc_fname)
        # Load the json
        dsc = json.load(open(dsc_path))
        # Validate the description
        try:
            label = dsc['meta']['clinical']['benign_malignant']
            if label not in {'benign', 'malignant'}:
                valid_dsc = False
        except KeyError:
            valid_dsc = False
        if not valid_dsc:
            # The description is invalid.
            # Mark it and it's corresponding image for moving
            img_name = dsc['name'] + '.jpg'
            image_path = join(imgs_dir, img_name)
            src_paths += [image_path, dsc_path]
            dst_paths += [join(inv_imgs_dir, img_name), join(inv_dscs_dir, dsc_fname)]

    # Move the invalid descriptions and images to the filtered directories
    multiprocessing.Pool(processes=num_processes).starmap(shutil.move, zip(src_paths, dst_paths))


def sort_images_into_class_and_learning_suites(images_dir, descs_dir, out_dir,
                                               train_port, val_port, num_processes):
    """
    Sorts the images in images_dir into classes, where each class
    will be stored in a directory by it's name.
    Than the method will sort the images into train val and test suites
    where the distribution of the classes will be the same in each suite.
    :param images_dir: Directory which holds images, and nothing else.
            Each image must have a corresponding description file in descs_dir.
    :param descs_dir: Directory which holds the descriptions of the images in image_dir,
            and nothing else.
    :param out_dir: Directory to store the sorting result.
    :param train_port: The portion of the images that should go into the train suite.
            should maintain 0 < x < 1.
    :param val_port: The portion of the images that should go into the val suite.
            should maintain 0 < x < 1.
    :param num_processes: Number of processes to split the work between
    """
    # Get the images paths and labels
    img_fnames, labels = get_image_names_labels(images_dir, descs_dir, num_processes)
    # Sort the images into classes directories
    classes_dir = join(out_dir, 'Class directories')
    sort_images_into_classes(imgs_dir=images_dir, img_fnames=img_fnames, labels=labels,
                             out_dir=classes_dir, num_processes=num_processes)
    # Sort the images into learning suites
    learning_suites_dir = join(out_dir, 'Learning suites')
    sort_images_into_learning_suites(src_dir=classes_dir, out_dir=learning_suites_dir,
                                     train_port=train_port, val_port=val_port, num_processes=num_processes)
    # Remove the (now empty) classes directories
    shutil.rmtree(classes_dir)


def get_image_names_labels(images_dir, descs_dir, num_processes):
    # Validate the inputs
    io.validate_exists_and_dir(images_dir, 'images_dir')
    io.validate_exists_and_dir(descs_dir, 'descs_dir')

    # Prepare a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Find all the description files
    desc_files = os.listdir(descs_dir)
    desc_paths = [join(descs_dir, desc_file) for desc_file in desc_files]
    # Read the name and label of the image inside each description file
    names, labels = zip(*pool.map(parse_names_labels, desc_paths))
    # Assert that all the images exist
    for name in names:
        image_path = join(images_dir, name + '.jpg')
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            raise ValueError("The image named {0} has a description file, but the image isn't found in images_dir {1}" \
                             .format(name, images_dir))

    # For each image name, get the corresponding image file name
    image_fnames = [name + '.jpg' for name in names]

    return image_fnames, labels


def sort_images_into_classes(imgs_dir, img_fnames, labels, out_dir, num_processes):
    """
    Sorts the images in images_dir into classes, where each class
    will be stored in a directory by it's name.
    :param imgs_dir: Directory which holds images, and nothing else.
                    Each image must have a corresponding description file in descs_dir.
    :param descs_dir: Directory which holds the descriptions of the images in image_dir,
                    and nothing else.
    :param out_dir: The directory in which to store the classes directories.
    """
    # Validate the inputs
    # Assert that all the image names point to an existing image
    for image_name in img_fnames:
        if not os.path.exists(join(imgs_dir, image_name)):
            raise ValueError("The image {0} does not exist in {1}".format(image_name, imgs_dir))

    # Create a processes poll
    pool = multiprocessing.Pool(processes=num_processes
                                )
    # Sort the image names into their classes by their corresponding labels:
    classes, names_sorted = sort_elements_into_classes(img_fnames, labels)
    # Create a directory for each class and copy all class images into it
    for klass in classes:
        class_dir = join(out_dir, klass)
        io.create_or_recreate_dir(class_dir)
        img_fnames = names_sorted[klass]
        # Copy the images from their src to dst
        src_image_paths = [join(imgs_dir, image_name) for image_name in img_fnames]
        dst_image_paths = [join(class_dir, image_name) for image_name in img_fnames]
        pool.starmap(shutil.copy, zip(src_image_paths, dst_image_paths))


def parse_names_labels(desc_path):
    """
    Parses the name and label of an image from it's description file
    :param desc_path: Path to the description file of the image
    :return: 2-tuple
    """
    desc_json = json.load(open(desc_path))
    name = desc_json["name"]
    label = desc_json["meta"]["clinical"]["benign_malignant"]
    return name, label


def sort_elements_into_classes(elements: list, labels: list):
    """
    Finds the existing classes among the elements
    and sorts the elements into classes.
    :param elements: List of elements.
    :param labels: List of labels corresponding to the elements.
    :return: The classes set of the elements, and the elements sorted by classes
    """
    # Validate the inputs
    assert len(elements) == len(labels), 'The number of elements should be equal to the number of labels'

    classes = set(labels)
    elements_sorted = {}
    for klass in classes:
        class_elements = [elem for elem, label in zip(elements, labels) if label == klass]
        elements_sorted[klass] = class_elements
    return classes, elements_sorted


def sort_images_into_learning_suites(src_dir, out_dir, train_port, val_port, num_processes):
    """
    Sorts the images into train, val and test suites,
    such that in each suite the distribution of classes will be the same.
    Each suite will be stored in a directory named after the suite.
    :param src_dir: Directory which contains images sorted into directories by their classes
    :param out_dir: Directory to hold the directories of the train, val and test suites.
    :param train_port: The portion of the images that should go into the train suite.
            should maintain 0 < x < 1.
    :param val_port: The portion of the images that should go into the val suite.
            should maintain 0 < x < 1.
    :raise AssertionError if the sum of train_port and val_port is greater or equal to 1.
    :raise RuntimeError if the size of class in some suite comes out 0 by the partition required.
    """
    # Validate the inputs
    io.validate_exists_and_dir(src_dir, 'src_dir')
    assert 0 < train_port < 1, "The train portion should maintain 0 < x < 1"
    assert 0 < val_port < 1, "The val portion should maintain 0 < x < 1"
    test_port = 1 - train_port - val_port
    assert 0 < test_port, "The sum of train and val portions should be below 1 to allow a test portion"

    suite_portions = {'train': train_port, 'val': val_port, 'test': test_port}

    # Prepare processes pool
    pool = multiprocessing.Pool(processes=num_processes)

    # Find all the classes in src_dir
    classes = os.listdir(src_dir)

    # Partition each class into train val and test suites
    sources = []
    dests = []
    for klass in classes:
        # Get all the images of the class
        image_names = os.listdir(join(src_dir, klass))
        image_paths = [join(src_dir, klass, image_name) for image_name in image_names]
        num_images = len(image_paths)
        # Partition the class into train val and test suites
        for suite in sorted(suite_portions):
            # Create the suite directory
            suite_dir = join(out_dir, suite, klass)
            io.create_or_recreate_dir(suite_dir)
            # Take the corresponding portion from the class
            suite_size = int(num_images * suite_portions[suite])
            if suite_size == 0:
                raise ValueError("The size of the {0} suite taken from class {1} comes out 0. "
                                 "Increase it's partition size or the number of images"
                                 .format(suite, klass))
            suite_images = image_paths[:suite_size]
            image_paths = image_paths[suite_size:]
            # Mark to move suite images into the suite directory
            for image_path in suite_images:
                sources.append(image_path)
                dests.append(suite_dir)

        # Due to the flooring in the calculation in the suite sizes,
        # There might be up to 3 images that were not placed in any suite.
        # Put the remaining images inside the train suite
        assert len(image_paths) <= 3
        for image_path in image_paths:
            shutil.move(image_path, join(out_dir, 'train', klass))

    # Move all the images marked earlier
    pool.starmap(shutil.move, zip(sources, dests))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='Directory which holds the images, and only them')
    parser.add_argument('descs_dir', type=str, help='Directory which holds the descriptions of the images, '
                                                    'and only them')
    parser.add_argument('dst_dir', type=str, help='Directory to store the sorting result')
    parser.add_argument('train_portion', type=float, help='Portion of the images to be in the train suite')
    parser.add_argument('val_portion', type=float, help='Portion of the images to be in the val suite')
    parser.add_argument('--p', type=int, help='Number of processes to use in parallel', default=16)
    args = parser.parse_args()

    inv_imgs_dir = join(args.dst_dir, 'Invalid Images')
    inv_dscs_dir = join(args.dst_dir, 'Invalid Descriptions')
    filter_invalid_images(imgs_dir=args.images_dir, dscs_dir=args.descs_dir,
                          inv_imgs_dir=inv_imgs_dir, inv_dscs_dir=inv_dscs_dir, num_processes=args.p)
    sort_images_into_class_and_learning_suites(images_dir=args.images_dir,
                                               descs_dir=args.descs_dir,
                                               out_dir=args.dst_dir,
                                               train_port=args.train_portion, val_port=args.val_portion,
                                               num_processes=args.p)
