import csv
import cv2
import numpy as np

from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras.models import Sequential
from os import listdir
from os.path import isdir, isfile, join

DATA_ROOT_PATH = './data'
DRIVING_LOG_FILE_NAME = 'driving_log.csv'
CORRECTION = 0.2


def read_data():
    """Read data from data directory."""
    images = []
    angles = []
    for sub_dir_name in listdir(DATA_ROOT_PATH):
        sub_dir = join(DATA_ROOT_PATH, sub_dir_name)
        if not isdir(sub_dir) or not isfile(join(sub_dir, DRIVING_LOG_FILE_NAME)):
            continue
        for image_center, image_left, image_right, angle in read_data_from_one_dir(sub_dir):
            images.extend([image_center, image_left, image_right, np.fliplr(
                image_center), np.fliplr(image_left), np.fliplr(image_right)])
            angles.extend([angle, angle + CORRECTION, angle - CORRECTION, -
                           angle, -angle - CORRECTION, -angle + CORRECTION])
    return np.array(images), np.array(angles)


def read_image(original_image_path):
    """Read image from a path"""
    image_partial_path = '/'.join(original_image_path.split('/')[-3:])
    current_path = join(DATA_ROOT_PATH, image_partial_path)
    image = cv2.imread(current_path)
    return cv2.imread(current_path)


def read_data_from_one_dir(data_dir):
    """There are multiple directories under data/, each subdirectory contains a driving_log.csv. This function reads
    from one of the sub-directores, given the directory path."""
    with open(join(data_dir, DRIVING_LOG_FILE_NAME)) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            yield read_image(line[0]), read_image(line[1]), read_image(line[2]), float(line[3])


def lenet():
    """Lenet architecture used at first, before using Nvidia architecture."""
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                         input_shape=(160, 320, 3)))  # 55x320x3
    model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Convolution2D(6, 5, 5, activation='relu'))  # 51x316x6
    model.add(MaxPooling2D())  # 25x153x6
    model.add(Convolution2D(16, 5, 5, activation='relu'))  # 21x149x16
    model.add(MaxPooling2D())  # 10x74x16
    model.add(Flatten())  # 11840
    model.add(Dense(1000))
    model.add(Dense(70))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def nvidia():
    """Network architecture by Nvidia."""
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                         input_shape=(160, 320, 3)))  # 65x320x3
    model.add(Lambda(lambda x: x / 255 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(
        2, 2), activation='relu'))  # 31x158x24
    model.add(Convolution2D(36, 5, 5, subsample=(
        2, 2), activation='relu'))  # 14x77x36
    model.add(Convolution2D(48, 5, 5, subsample=(
        2, 2), activation='relu'))  # 5x37x48
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 3x35x64
    model.add(Convolution2D(64, 3, 3, activation='relu'))  # 1x33x64
    model.add(Flatten())  # 2112
    model.add(Dense(200))  # 200
    model.add(Dense(14))  # 14
    model.add(Dense(1))  # 1
    model.compile(loss='mse', optimizer='adam')
    return model


if __name__ == '__main__':
    X_data, y_data = read_data()
    model = nvidia()
    model.fit(X_data, y_data, validation_split=0.2, shuffle=True, nb_epoch=5)
    print('Training complete, saving model')
    model.save('model.h5')
