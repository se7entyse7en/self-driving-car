import math
import os
import random
from functools import partial

import cv2

import pandas as pd

import keras.backend as K
import keras.losses

from matplotlib import pyplot as plt

from self_driving_car.dataset import preprocess_image


def fix_images_path_prefix(csv_path, to_path_prefix, to_csv_path=None):
    df = pd.read_csv(
        csv_path, header=None,
        names=('center_image_path', 'left_image_path', 'right_image_path',
               'steering_angle', 'speed', 'throttle', 'brake')
        )

    for col in ('center_image_path', 'left_image_path', 'right_image_path'):
        df[col] = df[col].apply(partial(fix_path_prefix,
                                        to_path_prefix=to_path_prefix))

    to_csv_path = to_csv_path or csv_path
    df.to_csv(to_csv_path, header=None)


def fix_path_prefix(image_path, to_path_prefix):
    rel_path = image_path[image_path.rindex('IMG'):]
    return os.path.join(to_path_prefix, rel_path)


def get_random_image_paths(dataset_path, n):
    base_path = os.path.join(dataset_path, 'IMG')
    all_images_paths = os.listdir(base_path)
    return [os.path.join(base_path, x)
            for x in random.sample(all_images_paths, n)]


def try_augmenter(augmenter, images_paths, augmentation_kwargs,
                  figsize=(12, 5), fontsize=16, preprocess=False, ncols=4):
    total_images = len(images_paths)
    ncols = min(ncols, total_images)
    nrows = math.ceil(total_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, (im_path, kwargs) in enumerate(
            zip(images_paths, augmentation_kwargs)):
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        if preprocess:
            im = preprocess_image(im)

        aug_im = augmenter.process(im, **kwargs)

        row, col = int(i / ncols), i % ncols
        ax = axes[row][col]

        kwargs_str = ','.join(f'{k}={v}' for k, v in kwargs.items())
        ax.set_title(f'Augmentation kwargs:\n{kwargs_str}',
                     fontdict={'fontsize': fontsize})
        ax.imshow(aug_im)

    fig.suptitle('Augmented images', fontsize=fontsize)
    plt.show()


def try_multi_augmenters(augmenters, images_paths, figsize=(12, 5),
                         fontsize=16, preprocess=False, ncols=4):
    total_images = len(images_paths)
    ncols = min(ncols, total_images)
    nrows = math.ceil(total_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, im_path in enumerate(images_paths):
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        if preprocess:
            im = preprocess_image(im)

        for aug in augmenters:
            kwargs = aug.gen_random_kwargs(im)
            im = aug.process(im, **kwargs)

        row, col = int(i / ncols), i % ncols
        ax = axes[row][col]

        ax.imshow(im)

    fig.suptitle('Augmented images', fontsize=fontsize)
    plt.show()


def mean_exponential_error(y_pred, y_true):
    return K.mean(K.exp(K.abs(y_pred - y_true)) - 1, axis=-1)


keras.losses.mean_exponential_error = mean_exponential_error
