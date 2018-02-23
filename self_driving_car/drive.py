import argparse
import base64
import os
import shutil
from datetime import datetime
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


sio = socketio.Server()
app = Flask(__name__)
model = None


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


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data['steering_angle']
        throttle = data['throttle']
        speed = data['speed']
        img_str = data['image']

        img_bytes = bytearray(BytesIO(base64.b64decode(img_str)).read())
        image = cv2.imdecode(np.asarray(img_bytes, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = preprocess_image(image)

        steering_angle = float(model.predict(np.array([image]), batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)

        send_control(steering_angle, throttle)

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help=('Path to image folder. This is where the images from the run '
              'will be saved.')
    )
    args = parser.parse_args()

    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print('Creating image folder at {}'.format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print('RECORDING THIS RUN ...')
    else:
        print('NOT RECORDING THIS RUN ...')

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
