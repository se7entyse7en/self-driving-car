import os
import shutil

import cv2

import numpy as np

import pandas as pd

from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter


IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64
CROP_TOP, CROP_BOTTOM = 30, 25


class DatasetHandler(object):

    COLUMNS = ('center', 'left', 'right', 'steering_angle', 'speed',
               'throttle', 'brake')
    TRANSFORMED_COLUMNS = ('pov', 'path', 'steering_angle')

    @classmethod
    def read(cls, *paths, transform=True):
        dataset = pd.concat([pd.read_csv(p, header=None, names=cls.COLUMNS)
                             for p in paths])
        if transform:
            dataset = pd.melt(dataset, id_vars=['steering_angle'],
                              value_vars=['center', 'left', 'right'],
                              var_name='pov', value_name='path')
        return dataset

    @classmethod
    def write(cls, df, path, transformed=True):
        cols = cls.TRANSFORMED_COLUMNS if transformed else cls.COLUMNS
        df.to_csv(path, index=False, header=False, columns=cols)


class DatasetGenerator(object):

    def __init__(self, training_set, validation_set, image_data_augmenters):
        self._training_set = training_set
        self._validation_set = validation_set
        self._augmenters = image_data_augmenters

    @classmethod
    def from_dataframe(cls, dataset, image_data_augmenters,
                       validation_size=0.25):
        validation_set = dataset.sample(frac=validation_size)
        training_set = dataset.iloc[~dataset.index.isin(
            validation_set.index)]

        return cls(training_set, validation_set, image_data_augmenters)

    @classmethod
    def shuffle_dataset(cls, dataset):
        return dataset.sample(frac=1).reset_index(drop=True)

    @property
    def training_set(self):
        return self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    def training_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._training_set, batch_size, self._augmenters)

    def validation_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._validation_set, batch_size)

    def _dataset_batch_generator(self, dataset, batch_size, augmenters=None):
        i = 0
        augmenters = augmenters or []
        batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                                dtype=np.uint8)
        batch_steerings = np.empty(batch_size)
        while True:
            for image, steering_angle in self._flow(
                    self.shuffle_dataset(dataset), augmenters):
                batch_images[i] = image
                batch_steerings[i] = steering_angle
                i += 1
                if i == batch_size:
                    yield batch_images, batch_steerings
                    i = 0

    def _flow(self, dataset, augmenters):
        for _, row in dataset.iterrows():
            yield self._flow_from_row(row, augmenters)

    def _flow_from_row(self, row, augmenters):
        image = preprocess_image_from_path(row['path'])
        if row.get('flip'):
            image = HorizontalFlipImageDataAugmenter.process(image)

        for aug in self._augmenters:
            image = aug.process_random(image)

        return image, row['steering_angle']


def preprocess_image_from_path(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return preprocess_image(image)


def preprocess_image(image):
    # Crop from bottom to remove car parts
    # Crop from top to remove part of the sky
    cropped_image = image[CROP_TOP:-CROP_BOTTOM, :]
    return cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv2.INTER_AREA)
