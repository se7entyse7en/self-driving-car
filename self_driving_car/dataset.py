import random

import cv2

import numpy as np

import pandas as pd

from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter


IMAGE_WIDTH, IMAGE_HEIGHT = 200, 66


class DatasetGenerator(object):

    def __init__(self, csv_path, image_data_augmenters,
                 data_augmenters_probs=None, test_size=0.25):
        self._csv_path = csv_path
        self._dataset = pd.read_csv(
            self._csv_path, header=None,
            names=('center_image_path', 'left_image_path', 'right_image_path',
                   'steering_angle', 'speed', 'throttle', 'brake')
        )
        shuffled_dataset = self._shuffle_dataset(self._dataset)

        n_rows = shuffled_dataset.shape[0]
        self._training_size = int(n_rows * (1 - test_size))
        self._test_size = n_rows - self._training_size

        self._training_set = shuffled_dataset.head(self._training_size)
        self._test_set = shuffled_dataset.tail(-self._test_size)

        self._augmenters = image_data_augmenters
        self._augmenters_probs = (data_augmenters_probs or
                                  [0.5] * len(self._augmenters))

    @property
    def training_size(self):
        return self._training_size

    @property
    def test_size(self):
        return self._test_size

    def flow(self, use_augmenters=True):
        for _, row in self._dataset.iterrows():
            yield from self._flow_from_row(row, use_augmenters)

    def training_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._training_set, batch_size, True)

    def test_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._test_set, batch_size, False)

    def preprocess_images(self, images_paths):
        # Only use center image for now
        yield preprocess_image_from_path(images_paths[0])

    def _shuffle_dataset(self, dataset):
        return dataset.sample(frac=1).reset_index(drop=True)

    def _flow_from_row(self, row, use_augmenters):
        images_paths = (row['center_image_path'],
                        row['left_image_path'],
                        row['right_image_path'])
        steering_angle = row['steering_angle']

        for image in self.preprocess_images(images_paths):
            if use_augmenters:
                for aug, aug_prob in zip(self._augmenters,
                                         self._augmenters_probs):
                    if random.random() > aug_prob:
                        continue

                    image, steering_angle = self._augment(
                        aug, image, steering_angle)

            yield image, steering_angle

    def _augment(self, augmenter, image, steering_angle):
        augmented_image = augmenter.process_random(image)
        if isinstance(augmenter, HorizontalFlipImageDataAugmenter):
            steering_angle = -steering_angle

        return augmented_image, steering_angle

    def _dataset_batch_generator(self, dataset, batch_size, use_augmenters):
        i = 0
        batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                                dtype=np.uint8)
        batch_steerings = np.empty(batch_size)
        while True:
            for _, row in self._shuffle_dataset(dataset).iterrows():
                for image, steering_angle in self._flow_from_row(
                        row, use_augmenters):
                    batch_images[i] = image
                    batch_steerings[i] = steering_angle
                    i += 1
                    if i == batch_size:
                        yield batch_images, batch_steerings
                        i = 0


def preprocess_image_from_path(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return preprocess_image(image)


def preprocess_image(image):
    # Crop 25 pixels from bottom to remove car parts
    cropped_image = image[:-25, :]
    return cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv2.INTER_AREA)
