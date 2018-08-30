import math
import os
import random
import subprocess
import tempfile
from datetime import datetime
from functools import partial

import cv2

import pandas as pd

import keras.backend as K
import keras.losses

import matplotlib.pyplot as plt
import seaborn as sns

from self_driving_car.dataset import DatasetHandler
from self_driving_car.dataset import preprocess_image


sns.set()


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


def plot_steerings_distribution(steering_angles, bins=None):
    bins = bins or [-1.0, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1.0]
    p = sns.distplot(steering_angles, bins=bins, kde=False)
    p.set_xticks(bins)
    plt.show()


def build_video_from_dataset(dataset_csv_path, output, steering_overlay=True,
                             speed_modifier=1):

    def add_steering_overlay(target_dir, orig_image_path, steering_angle):
        image = cv2.imread(orig_image_path)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'{steering_angle:+5.3f}'
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        textX = (image.shape[1] - textsize[0]) // 2
        textY = image.shape[0] - textsize[1] - 10

        cv2.putText(image, text, (textX, textY), font, 1, (0, 255, 0), 2)

        output_path = os.path.join(
            target_dir, os.path.split(orig_image_path)[1])

        cv2.imwrite(output_path, image)

        return output_path

    def extract_time(item):
        filename = os.path.split(item.center)[1]
        return datetime.strptime(filename[7:-4], '%Y_%m_%d_%H_%M_%S_%f')

    df = DatasetHandler.read(dataset_csv_path, transform=False)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        with open(os.path.join(
                tmp_dir_name, 'concat_demuxer.txt'), 'w') as fout:
            iterator = df.itertuples()
            curr_item = next(iterator)
            curr_time = extract_time(curr_item)
            for next_item in iterator:
                image_file_name = add_steering_overlay(
                    tmp_dir_name, next_item.center,
                    next_item.steering_angle)

                next_time = extract_time(next_item)
                duration = ((next_time - curr_time).total_seconds() *
                            speed_modifier)

                fout.write(f'file \'{image_file_name}\'\n')
                fout.write(f'duration {duration}\n')

                curr_item, curr_time = next_item, next_time

        cmd = [
            'ffmpeg',
            '-f',
            'concat',
            '-safe',
            '0',
            '-i',
            fout.name,
            '-vsync',
            'vfr',
            '-pix_fmt',
            'yuv420p',
            output
        ]

        subprocess.run(cmd)


keras.losses.mean_exponential_error = mean_exponential_error
