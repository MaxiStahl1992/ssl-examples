import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def load_cifar_batch(batch_filename):
    with open(batch_filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels

def load_cifar10_data(data_dir='../data/cifar-10-batches-py'):
    x_train, y_train = [], []
    for i in range(1, 6):
        features, labels = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(features)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model(input_shape=(32, 32, 3)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    return model

def generate_pseudo_labels(model, data, batch_size=64):
    pseudo_labels = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_pseudo_labels = model.predict(batch_data)
        batch_pseudo_labels = np.argmax(batch_pseudo_labels, axis=1)
        pseudo_labels.append(batch_pseudo_labels.numpy())
    return np.concatenate(pseudo_labels)

def get_data_generator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1):
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=True
    )
   