import argparse
import base64
import os
import shutil
from datetime import datetime
from functools import partial
from io import BytesIO

import eventlet
import eventlet.wsgi
import h5py
import socketio
from flask import Flask

import cv2

import numpy as np

from keras import __version__ as keras_version
from keras.models import load_model

from self_driving_car.dataset import preprocess_image


app = Flask(__name__)


class SimplePIController(object):

    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        self.error = self.set_point - measurement
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


class Handlers(object):

    @staticmethod
    def telemetry(sid, data, model, img_folder):
        if data:
            steering_angle = data['steering_angle']
            throttle = data['throttle']
            speed = data['speed']
            img_str = data['image']

            img_bytes = bytearray(BytesIO(base64.b64decode(img_str)).read())
            image = cv2.imdecode(np.asarray(img_bytes, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)

            steering_angle = float(model.predict(
                np.array([image]), batch_size=1))

            throttle = controller.update(float(speed))

            print(f'Predicted: {steering_angle}, {throttle}')

            send_control(steering_angle, throttle)

            if img_folder != '':
                timestamp = datetime.utcnow().strftime(
                    '%Y_%m_%d_%H_%M_%S_%f')[:-3]
                image_filename = os.path.join(
                    img_folder, timestamp) + '.png'
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_filename, image)
        else:
            # NOTE: DON'T EDIT THIS.
            sio.emit('manual', data={}, skip_sid=True)

    @staticmethod
    def connect(sid, environ):
        send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer',
             data={'steering_angle': str(steering_angle),
                   'throttle': str(throttle)},
             skip_sid=True)


def load_keras_model(model):
    f = h5py.File(model, mode='r')
    model_version = f.attrs.get('keras_version')

    if model_version != keras_version:
        print(f'You are using Keras version {keras_version}, '
              f'but the model was built using {model_version}')

    return load_model(model)


def setup_image_folder(img_folder):
    if img_folder != '':
        print(f'Creating image folder at {img_folder}')
        if os.path.exists(img_folder):
            shutil.rmtree(img_folder)
        os.makedirs(img_folder)
        print('RECORDING THIS RUN ...')
    else:
        print('NOT RECORDING THIS RUN ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model', type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--image_folder', type=str, default='',
        help=('Path to image folder. This is where the images from the run '
              'will be saved.')
    )
    parser.add_argument(
        '--speed', type=float, default=10.0,
        help='Desired speed.'
    )
    args = parser.parse_args()

    model = load_keras_model(args.model)
    setup_image_folder(args.image_folder)

    controller = SimplePIController(0.1, 0.002)
    controller.set_desired(args.speed)

    sio = socketio.Server()
    # Attach handlers
    sio.on('telemetry', partial(
        Handlers.telemetry, model=model, img_folder=args.image_folder))
    sio.on('connect', Handlers.connect)

    # Wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
