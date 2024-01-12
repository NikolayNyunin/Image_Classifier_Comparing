from time import time

from tensorflow import keras
import matplotlib.pyplot as plt

from data_loading import load_data


def train_and_evaluate(dataset_name: str, train_percentage: int, test_percentage: int, epoch_count: int,
                       batch_size: int, model_tuples: list[tuple]) -> (list, list, plt.Figure):
    """Обучение и тестирование моделей."""

    # загрузка данных
    (train_images, train_labels), (test_images, test_labels)\
        = load_data(dataset_name, train_percentage, test_percentage)

    # создание графика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    metrics, train_times = [], []
    for model_name, model_function in model_tuples:
        # создание экземпляра модели
        model: keras.Sequential = model_function()

        start_time = time()

        # обучение модели
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(test_images, test_labels),
            epochs=epoch_count,
            batch_size=batch_size
        )

        train_time = time() - start_time

        metrics.append(model.evaluate(test_images, test_labels))
        train_times.append(train_time)

        ax1.plot(history.history['val_accuracy'], '-o', label=model_name)
        ax2.plot(history.history['val_loss'], '-o', label=model_name)

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Epoch')
    ax1.set_xticks(range(epoch_count))
    ax1.legend()

    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_xticks(range(epoch_count))
    ax2.legend()

    return metrics, train_times, fig
