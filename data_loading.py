import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_data(dataset_name: str, train_percentage: int, test_percentage: int)\
        -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """Загрузка и первичная обработка данных."""

    if dataset_name == 'MNIST':
        dataset = keras.datasets.mnist
    elif dataset_name == 'Fashion-MNIST':
        dataset = keras.datasets.fashion_mnist
    elif dataset_name == 'CIFAR-10':
        dataset = keras.datasets.cifar10
    else:
        raise ValueError('`dataset_name` is not recognized: {}'.format(dataset_name))

    # загрузка изображений из датасета
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # вычисление количества используемых изображений
    n_train = round(train_images.shape[0] * (train_percentage / 100))
    n_test = round(test_images.shape[0] * (test_percentage / 100))

    # ограничение размера датасета
    train_images: np.ndarray = train_images[:n_train]
    train_labels: np.ndarray = train_labels[:n_train]
    test_images: np.ndarray = test_images[:n_test]
    test_labels: np.ndarray = test_labels[:n_test]

    # нормализация изображений из диапазона [0, 255] в [0, 1]
    train_images = train_images / 255
    test_images = test_images / 255

    # добавление дополнительной размерности к данным для свёрточных каналов
    train_images = np.array(train_images[..., tf.newaxis])
    test_images = np.array(test_images[..., tf.newaxis])

    return (train_images, train_labels), (test_images, test_labels)
