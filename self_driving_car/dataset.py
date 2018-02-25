import cv2

import numpy as np

import pandas as pd

from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter


IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64
CROP_TOP, CROP_BOTTOM = 50, 25


class DatasetGenerator(object):

    def __init__(self, training_set, test_set, image_data_augmenters):
        self._training_set = training_set
        self._test_set = test_set
        self._augmenters = image_data_augmenters

    @classmethod
    def from_csv(cls, csv_path, image_data_augmenters, test_size=0.25):
        dataset = pd.read_csv(
            csv_path, header=None,
            names=('center_image_path', 'left_image_path', 'right_image_path',
                   'steering_angle', 'speed', 'throttle', 'brake')
        )
        shuffled_dataset = cls.shuffle_dataset(dataset)

        n_rows = shuffled_dataset.shape[0]
        training_size = int(n_rows * (1 - test_size))
        test_size = n_rows - training_size

        training_set = shuffled_dataset.head(training_size)
        test_set = shuffled_dataset.tail(-test_size)

        return cls(training_set, test_set, image_data_augmenters)

    @classmethod
    def shuffle_dataset(cls, dataset):
        return dataset.sample(frac=1).reset_index(drop=True)

    def flow(self, use_augmenters=True):
        for _, row in self._dataset.iterrows():
            yield from self._flow_from_row(row, use_augmenters)

    def training_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._training_set, batch_size, True)

    def test_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._test_set, batch_size, False)

    def _flow_from_row(self, row, use_augmenters):
        steering_angle = row['steering_angle']

        images = {
            'center': preprocess_image_from_path(row['center_image_path']),
            'left': preprocess_image_from_path(row['left_image_path']),
            'right': preprocess_image_from_path(row['right_image_path']),
        }

        for pov, image in images.items():
            if use_augmenters:
                for aug in self._augmenters:
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
    # Crop from bottom to remove car parts
    # Crop from top to remove part of the sky
    cropped_image = image[CROP_TOP:-CROP_BOTTOM, :]
    return cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT),
                      interpolation=cv2.INTER_AREA)
