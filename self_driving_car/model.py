from keras import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    model.compile(loss='mse', optimizer=sgd, metrics=['mae'])

    return model


def load_data_generator(csv_path, test_size=0.25, use_center_only=False):
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
    return DatasetGenerator.from_csv(csv_path, augmenters, test_size=test_size,
                                     use_center_only=use_center_only)


def train_model(model, csv_path, batch_size, epochs, test_size=0.25,
                use_center_only=False, use_augmenters=True,
                use_steering_correction=True, plot_history=False,
                plot_output_file=None, **kwargs):
    data_generator = load_data_generator(csv_path, test_size=test_size,
                                         use_center_only=use_center_only)
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5')

    training_set_gen = data_generator.training_set_batch_generator(
        batch_size, use_augmenters=use_augmenters,
        use_steering_correction=use_steering_correction)
    test_set_gen = data_generator.test_set_batch_generator(batch_size)

    steps_per_epoch = kwargs.get(
        'steps_per_epoch', int(
            data_generator.training_set.shape[0] / batch_size)
    )
    validation_steps = kwargs.get(
        'validation_steps', int(data_generator.test_set.shape[0] / batch_size)
    )
    history = model.fit_generator(
        training_set_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint],
        validation_data=test_set_gen,
        validation_steps=validation_steps,
    )

    if plot_history:
        plot_training_history(history, plot_output_file=plot_output_file,
                              **kwargs.get('plot_savefig_kwargs', {}))


def plot_training_history(history, figsize=(12, 5), plot_output_file=None,
                          **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _subplot_training_history(history, axes[0], 'Epoch', 'MAE',
                              'Model MAE', 'mean_absolute_error',
                              'val_mean_absolute_error')
    _subplot_training_history(history, axes[1], 'Epoch', 'Loss (MSE)',
                              'Model loss (MSE)', 'loss', 'val_loss')

    fig.suptitle('Training history', fontsize=16)

    if plot_output_file is None:
        plt.show()
    else:
        plt.savefig(plot_output_file, **kwargs)


def _subplot_training_history(history, ax, x_label, y_label, title,
                              train_metric, test_metric):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.plot(history.history[train_metric])
    ax.plot(history.history[test_metric])

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(['train', 'test'], loc='upper left')
