import os
import random

import cv2
import pandas as pd
from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter


class DatasetGenerator(object):

    def __init__(self, csv_path, image_data_augmenters,
                 data_augmenters_probs=None):
        self._csv_path = csv_path
        self._base_dataset = pd.read_csv(
            self._csv_path, header=None,
            names=('center_image_path', 'left_image_path', 'right_image_path',
                   'steering_angle', 'speed', 'throttle', 'brake')
        )
        self._augmenters = image_data_augmenters
        self._augmenters_probs = (data_augmenters_probs or
                                  [0.5] * len(self._augmenters))

    def flow(self, use_augmenters=True):
        for _, row in self._base_dataset.iterrows():
            yield from self._flow_from_row(row, use_augmenters)

    def preprocess_images(self, images_paths):
        # Only use center image for now
        yield preprocess_image(images_paths[0])

    def save(self, csv_output_path):
        out_rows = []
        img_dir = os.path.join(os.path.split(csv_output_path)[0], 'IMG')
        os.makedirs(img_dir)
        for i, (_, row) in enumerate(self._base_dataset.iterrows()):
            for j, (image, steering_angle) in enumerate(
                    self._flow_from_row(row, True)):
                image_path = os.path.join(img_dir, f'{i}-{j}.jpg')
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                out_rows.append({
                    'image_path': image_path,
                    'steering_angle': steering_angle
                })

        df = pd.DataFrame.from_records(out_rows)
        df.to_csv(csv_output_path)

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


def preprocess_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Crop 25 pixels from bottom to remove car parts
    cropped_image = image[:-25, :]
    return cv2.resize(cropped_image, (200, 66), interpolation=cv2.INTER_AREA)
