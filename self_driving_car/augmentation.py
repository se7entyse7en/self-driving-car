import cv2
import numpy as np


class ImageDataAugmenter(object):

    def generate(self, image, args_kwargs):
        for args, kwargs in args_kwargs:
            yield self.process(image, *args, **kwargs)

    def process(self, image, *args, **kwargs):
        raise NotImplementedError()


class HorizontalFlipImageDataAugmenter(ImageDataAugmenter):

    def process(self, image):
        return cv2.flip(image, 1)


class BrightnessImageDataAugmenter(ImageDataAugmenter):

    def process(self, image, perc):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv_image)

        V = V.astype('float64')
        V_modifier = V * perc / 100
        V_modified = np.clip(V + V_modifier, 0, 255).astype('uint8')

        image = cv2.merge([H, S, V_modified])
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


class HueImageDataAugmenter(ImageDataAugmenter):

    def process(self, image, angle):
        if angle < -180 or angle > 180:
            raise ValueError('Angle must be in range (-180, 180)')

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv_image)

        # The value of H is halved to fit in [0, 255]. See link:
        # https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        H = H.astype('float64')
        H_modified = ((H * 2 + angle) % 360 / 2).astype('uint8')
        image = cv2.merge([H_modified, S, V])
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


class SaturationImageDataAugmenter(ImageDataAugmenter):

    def process(self, image, perc):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv_image)

        S = S.astype('float64')
        S_modifier = S * perc / 100
        S_modified = np.clip(S + S_modifier, 0, 255).astype('uint8')

        image = cv2.merge([H, S_modified, V])
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)