import os
from functools import partial

import pandas as pd


def fix_images_path_prefix(csv_path, to_path_prefix, to_csv_path=None):
    df = pd.read_csv(
        csv_path, header=None,
        names=('center_image_path', 'left_image_path', 'right_image_path',
               'steering_angle', 'speed', 'throttle', 'brake')
        )

    for col in ('center_image_path', 'left_image_path', 'right_image_path'):
        df[col] = df[col].apply(partial(fix_path_prefix,
                                        to_path_prefix=to_path_prefix))

    to_csv_path = to_csv_path or csv_path
    df.to_csv(to_csv_path, header=None)


def fix_path_prefix(image_path, to_path_prefix):
    rel_path = image_path[image_path.rindex('IMG'):]
    return os.path.join(to_path_prefix, rel_path)
