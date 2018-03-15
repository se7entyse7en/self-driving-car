import cv2

import numpy as np

import pandas as pd

from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter


IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64
CROP_TOP, CROP_BOTTOM = 30, 25
STEERING_CORRECTION = {
    'left': 0.25,
    'center': 0,
    'right': -0.25
}


class DatasetGenerator(object):

    def __init__(self, training_set, test_set, image_data_augmenters):
        self._training_set = training_set
        self._test_set = test_set
        self._augmenters = image_data_augmenters

    @classmethod
    def from_csv(cls, csv_path, image_data_augmenters, test_size=0.25,
                 use_center_only=False):
        dataset = pd.read_csv(
            csv_path, header=None,
            names=('center', 'left', 'right', 'steering_angle',
                   'speed', 'throttle', 'brake')
        )
        dataset = pd.melt(dataset, id_vars=['steering_angle'],
                          value_vars=['center', 'left', 'right'],
                          var_name='pov', value_name='path')

        center_only = dataset[dataset.pov == 'center']
        not_center_only = dataset[dataset.pov != 'center']

        test_set = center_only.sample(frac=test_size)
        training_set = center_only.iloc[~center_only.index.isin(
            test_set.index)]
        if not use_center_only:
            training_set = pd.concat([training_set, not_center_only])

        return cls(training_set, test_set, image_data_augmenters)

    @classmethod
    def shuffle_dataset(cls, dataset):
        return dataset.sample(frac=1).reset_index(drop=True)

    @property
    def training_set(self):
        return self._training_set

    @property
    def test_set(self):
        return self._test_set

    def training_set_batch_generator(self, batch_size,
                                     use_augmenters=True,
                                     use_steering_correction=True):
        yield from self._dataset_batch_generator(
            self._training_set, batch_size, use_augmenters,
            use_steering_correction)

    def test_set_batch_generator(self, batch_size):
        yield from self._dataset_batch_generator(
            self._test_set, batch_size, False, False)

    def _dataset_batch_generator(self, dataset, batch_size, use_augmenters,
                                 use_steering_correction):
        i = 0
        batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3],
                                dtype=np.uint8)
        batch_steerings = np.empty(batch_size)
        while True:
            for image, steering_angle in self._flow(
                    self.shuffle_dataset(dataset), use_augmenters,
                    use_steering_correction):
                batch_images[i] = image
                batch_steerings[i] = steering_angle
                i += 1
                if i == batch_size:
                    yield batch_images, batch_steerings
                    i = 0

    def _flow(self, dataset, use_augmenters, use_steering_correction):
        for _, row in dataset.iterrows():
            yield self._flow_from_row(row, use_augmenters,
                                      use_steering_correction)

    def _flow_from_row(self, row, use_augmenters, use_steering_correction):
        image = preprocess_image_from_path(row['path'])
        steering_angle = row['steering_angle']

        if use_steering_correction:
            steering_angle += STEERING_CORRECTION[row['pov']]

        if use_augmenters:
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
