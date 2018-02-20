from keras import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda

from self_driving_car.augmentation import BlurringImageDataAugmenter
from self_driving_car.augmentation import BrightnessImageDataAugmenter
from self_driving_car.augmentation import HorizontalFlipImageDataAugmenter
from self_driving_car.augmentation import HueImageDataAugmenter
from self_driving_car.augmentation import ReflectionImageDataAugmenter
from self_driving_car.augmentation import RotationImageDataAugenter
from self_driving_car.augmentation import SaturationImageDataAugmenter
from self_driving_car.augmentation import ShadowImageDataAugmenter
from self_driving_car.augmentation import VerticalShiftImageDataAugmenter
from self_driving_car.dataset import DatasetGenerator


def build_model(input_shape, sgd_optimizer_params):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))

    model.add(Conv2D(3, (1, 1)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    sgd = optimizers.SGD(*sgd_optimizer_params)
    model.compile(loss='mse', optimizer=sgd)

    return model


def load_data_generator(csv_path, augmenters_probs=None, test_size=0.25):
    augmenters = [
        BlurringImageDataAugmenter(),
        BrightnessImageDataAugmenter(),
        HorizontalFlipImageDataAugmenter(),
        HueImageDataAugmenter(),
        ReflectionImageDataAugmenter(),
        RotationImageDataAugenter(),
        SaturationImageDataAugmenter(),
        ShadowImageDataAugmenter(),
        VerticalShiftImageDataAugmenter(),
    ]
    return DatasetGenerator(csv_path, augmenters,
                            data_augmenters_probs=augmenters_probs,
                            test_size=test_size)


def train_model(model, csv_path, batch_size, augmenters_probs=None,
                test_size=0.25, **kwargs):
    data_generator = load_data_generator(csv_path, augmenters_probs, test_size)
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', save_best_only=True)

    training_set_gen = data_generator.training_set_batch_generator(batch_size)
    test_set_gen = data_generator.test_set_batch_generator(batch_size)
    model.fit_generator(
        training_set_gen,
        steps_per_epoch=int(data_generator.training_size / batch_size),
        epochs=1,
        verbose=1,
        callbacks=[checkpoint],
        validation_data=test_set_gen,
        validation_steps=int(data_generator.test_size / batch_size),
    )
