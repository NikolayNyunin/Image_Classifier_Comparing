from inspect import getmembers, isfunction

from tensorflow import keras
import matplotlib.pyplot as plt

from data_loading import load_data
import models


def main():
    # задание параметров для обучения моделей
    dataset_name = 'Fashion-MNIST'
    train_percentage = 1
    test_percentage = 1

    epoch_count = 10
    batch_size = 4

    # получение информации о моделях, описанных в файле `models.py`
    model_tuples = getmembers(models, isfunction)

    # загрузка данных
    (train_images, train_labels), (test_images, test_labels)\
        = load_data(dataset_name, train_percentage, test_percentage)

    # создание графика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    trained_models = []
    histories = []
    for model_name, model_function in model_tuples:
        # создание экземпляра модели
        model: keras.Sequential = model_function()

        # обучение модели
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(test_images, test_labels),
            epochs=epoch_count,
            batch_size=batch_size
        )

        trained_models.append(model)
        histories.append(history)

        ax1.plot(history.history['val_accuracy'], label=model_name)
        ax2.plot(history.history['val_loss'], label=model_name)

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Epoch')
    ax1.set_xticks(range(epoch_count))
    ax1.legend()

    ax2.set_ylabel('Loss')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.set_xticks(range(epoch_count))
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
