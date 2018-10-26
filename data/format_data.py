import auxilleries
from auxilleries import MultiProcessing

import os
from os.path import join
import json
import shutil
import argparse
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from sklearn.model_selection import train_test_split
from PIL import Image


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
    auxilleries.IO.validate_exists_and_dir(imgs_dir, 'imgs_dir')
    auxilleries.IO.validate_exists_and_dir(dscs_dir, 'dscs_dir')
    # Create the result dirs
    auxilleries.IO.create_or_recreate_dir(inv_imgs_dir)
    auxilleries.IO.create_or_recreate_dir(inv_dscs_dir)

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
    p = multiprocessing.Pool(processes=num_processes)
    list(tqdm(p.imap(MultiProcessing.imap_wrapper, zip(repeat(shutil.move), src_paths, dst_paths)),
              total=len(src_paths), desc='Filtering invalid images'))


def get_image_names_labels(images_dir, descs_dir, num_processes):
    # Validate the inputs
    auxilleries.IO.validate_exists_and_dir(images_dir, 'images_dir')
    auxilleries.IO.validate_exists_and_dir(descs_dir, 'descs_dir')

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


def sort_images_into_suites(src_dir, out_dir, suites, h, w, num_processes):
    """
    Sorts the images into train, val and test suites,
    such that in each suite the distribution of classes will be the same.
    Each suite will be stored in a directory named after the suite.
    :param src_dir: Directory which contains images sorted into directories by their classes
    :param out_dir: Directory to hold the directories of the train, val and test suites.
    """
    sources = []
    dests = []
    for _class in suites.keys():
        for suite_name, fnames in suites[_class].items():
            suite_dir = join(out_dir, suite_name, _class)
            auxilleries.IO.create_or_recreate_dir(suite_dir)
            sources += [join(src_dir, fname) for fname in fnames]
            dests += [join(suite_dir, fname) for fname in fnames]

    # Copy the images into their suite directory
    pool = multiprocessing.Pool(processes=num_processes)
    list(tqdm(pool.imap(MultiProcessing.imap_wrapper, zip(repeat(resize_image), sources, dests, repeat(h), repeat(w))),
              total=len(sources), desc='Splitting to suites & Resizing'))


def resize_image(src, dst, h, w):
    try:
        img = Image.open(src).resize((w, h))
        img.save(dst)
    except OSError as e:
        if str(e) == 'cannot write mode RGBA as JPEG':
            # The image is actually RGBA and not JPEG.
            # Convert it to RGB, and while writing convert it to jpg
            img.convert('RGB').save(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', type=str, help='Directory which holds the images, and only them')
    parser.add_argument('descs_dir', type=str, help='Directory which holds the descriptions of the images, '
                                                    'and only them')
    parser.add_argument('dst_dir', type=str, help='Directory to store the sorting result')
    parser.add_argument('train_portion', type=float, help='Portion of the images to be in the train suite')
    parser.add_argument('val_portion', type=float, help='Portion of the images to be in the val suite')
    parser.add_argument('--h', type=int, help='Reqested height of images', default=229)
    parser.add_argument('--w', type=int, help='Reqested width of images', default=229)
    parser.add_argument('--p', type=int, help='Number of processes to use in parallel', default=4)
    args = parser.parse_args()

    inv_imgs_dir = join(args.dst_dir, 'Invalid Images')
    inv_dscs_dir = join(args.dst_dir, 'Invalid Descriptions')

    filter_invalid_images(imgs_dir=args.images_dir, dscs_dir=args.descs_dir,
                          inv_imgs_dir=inv_imgs_dir, inv_dscs_dir=inv_dscs_dir, num_processes=args.p)

    # Sort images into classes and suites
    out_dir = join(args.dst_dir, 'suites')
    # Get the images paths and labels
    img_fnames, labels = get_image_names_labels(args.images_dir, args.descs_dir, args.p)
    # Sort the images into classes directories
    classes, fnames_sorted = sort_elements_into_classes(img_fnames, labels)
    # Split the images to suites
    suites = {_class: None for _class in classes}
    for _class in classes:
        train, val_test = train_test_split(fnames_sorted[_class], train_size=args.train_portion)
        val, test = train_test_split(val_test, train_size=args.val_portion / (1 - args.train_portion))
        suites[_class] = {'train': train, 'val': val, 'test': test}
    # Move the images into suites directories
    sort_images_into_suites(src_dir=args.images_dir, out_dir=out_dir,
                            suites=suites, h=args.h, w=args.w, num_processes=args.p)
