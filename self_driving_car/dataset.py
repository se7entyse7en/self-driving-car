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


class DatasetPreprocessor(object):

    @classmethod
    def strip_straight(cls, input_csv_path, output_path,
                       straight_threshold=0.1):
        dataset = DatasetHandler.read(input_csv_path, transform=False)

        dataset = dataset[dataset.steering_angle.abs() > straight_threshold]
        dataset = cls._copy_images(dataset, output_path)

        DatasetHandler.write(
            dataset, os.path.join(output_path, 'driving_log.csv'),
            transformed=False
        )
        return dataset

    @classmethod
    def _copy_images(cls, dataset, output_path):

        def build_target_path(orig_path):
            return os.path.join(
                output_path, 'IMG', os.path.split(orig_path)[1])

        def copy_images(row):
            shutil.copy(row.center, row.center_target_path)
            shutil.copy(row.left, row.left_target_path)
            shutil.copy(row.right, row.right_target_path)

        os.makedirs(os.path.join(output_path, 'IMG'))

        extra_cols = ('center_target_path',
                      'left_target_path',
                      'right_target_path')
        dataset = dataset.apply(
            lambda r: pd.Series(
                [r.center, r.left, r.right, r.steering_angle, r.speed,
                 r.throttle, r.brake, build_target_path(r.center),
                 build_target_path(r.left), build_target_path(r.right)],
                index=DatasetHandler.COLUMNS + extra_cols), axis=1
        )

        dataset.apply(copy_images, axis=1)

        dataset['center'] = dataset['center_target_path']
        dataset['left'] = dataset['left_target_path']
        dataset['right'] = dataset['right_target_path']
        return dataset[list(DatasetHandler.COLUMNS)]


class DatasetGenerator(object):

    def __init__(self, training_set, validation_set, image_data_augmenters,
                 steering_correction=None):
        self._training_set = training_set
        self._validation_set = validation_set
        self._augmenters = image_data_augmenters
        if steering_correction:
            steer_corr = {
                'left': abs(steering_correction),
                'center': 0,
                'right': -abs(steering_correction)
            }
        else:
            steer_corr = None
        self._steering_correction = steer_corr

    @classmethod
    def from_csv(cls, csv_paths, image_data_augmenters, validation_size=0.25,
                 use_center_only=False, steering_correction=None):
        return cls.from_dataframe(
            image_data_augmenters, DatasetHandler.read(*csv_paths),
            validation_size=validation_size, use_center_only=use_center_only,
            steering_correction=steering_correction
        )

    @classmethod
    def from_dataframe(cls, dataset, image_data_augmenters,
                       validation_size=0.25, use_center_only=False,
                       steering_correction=None):
        center_only = dataset[dataset.pov == 'center']
        not_center_only = dataset[dataset.pov != 'center']

        validation_set = center_only.sample(frac=validation_size)
        training_set = center_only.iloc[~center_only.index.isin(
            validation_set.index)]
        if not use_center_only:
            training_set = pd.concat([training_set, not_center_only])

        return cls(training_set, validation_set, image_data_augmenters,
                   steering_correction=steering_correction)

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
        steering_angle = row['steering_angle']

        if self._steering_correction:
            steering_angle += self._steering_correction[row['pov']]

        for aug in self._augmenters:
            image, steering_angle = self._augment(
                aug, image, steering_angle)

        return image, steering_angle

    def _augment(self, augmenter, image, steering_angle):
        augmented_image = augmenter.process_random(image)
        if isinstance(augmenter, HorizontalFlipImageDataAugmenter):
            steering_angle = -steering_angle

        return augmented_image, steering_angle


def preprocess_image_from_path(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return preprocess_image(image)


def preprocess_image(image):
    # Crop from bottom to remove car parts
    # Crop from top to remove part of the sky
    cropped_image = image[CROP_TOP:-CROP_BOTTOM, :]
    return cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv2.INTER_AREA)
