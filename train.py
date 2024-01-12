from threading import Thread
from time import time

from tensorflow import keras
import matplotlib.pyplot as plt

from data_loading import load_data


class TrainAndEvaluateThread(Thread):
    """Класс потока обучения и оценки моделей."""

    def __init__(self, dataset_name: str, train_percentage: int, test_percentage: int,
                 epoch_count: int, batch_size: int, model_tuples: list[tuple]):
        super().__init__()

        self.dataset_name = dataset_name
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.model_tuples = model_tuples

        self.metrics, self.train_times, self.figure = None, None, None

    def run(self):
        """Обучение и тестирование моделей."""

        # загрузка данных
        (train_images, train_labels), (test_images, test_labels)\
            = load_data(self.dataset_name, self.train_percentage, self.test_percentage)

        # создание графика
        self.figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        self.metrics, self.train_times = [], []
        for model_name, model_function in self.model_tuples:
            print('-----Модель "{}"-----'.format(model_name))

            # создание экземпляра модели
            model: keras.Sequential = model_function()

            start_time = time()

            # обучение модели
            history = model.fit(
                train_images,
                train_labels,
                validation_data=(test_images, test_labels),
                epochs=self.epoch_count,
                batch_size=self.batch_size,
                verbose=2
            )

            train_time = time() - start_time

            self.metrics.append(model.evaluate(test_images, test_labels, verbose=0))
            self.train_times.append(train_time)

            ax1.plot(history.history['val_accuracy'], '-o', label=model_name)
            ax2.plot(history.history['val_loss'], '-o', label=model_name)

            print()

        ax1.set_ylabel('Accuracy')
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Epoch')
        ax1.legend()

        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()

        if self.epoch_count <= 20:
            ax1.set_xticks(range(self.epoch_count))
            ax2.set_xticks(range(self.epoch_count))
