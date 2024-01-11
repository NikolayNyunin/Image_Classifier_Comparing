import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror

from train import train_and_evaluate

DATASETS = ('MNIST', 'Fashion-MNIST', 'CIFAR-10')


class App(tk.Tk):
    """Основной класс приложения."""

    def __init__(self):
        super().__init__()

        self.title('Сравнение классификаторов изображений')
        self.geometry('480x360')
        self.resizable(False, False)


class ParameterSelectionFrame(ttk.Frame):
    """Класс, описывающий виджет выбора параметров для тестирования моделей."""

    def __init__(self, master, model_tuples):
        super().__init__(master)

        self.model_tuples = model_tuples
        model_names = [name for name, _ in model_tuples]

        padding = {'padx': 5, 'pady': 5}

        self.model_label_frame = ttk.LabelFrame(self, text='Модели для тестирования:')

        self.selected_models = [tk.BooleanVar(value=True) for _ in range(len(model_names))]
        for i, model_name in enumerate(model_names):
            checkbox = ttk.Checkbutton(
                self.model_label_frame,
                text=model_name,
                variable=self.selected_models[i],
                onvalue=True,
                offvalue=False
            )
            checkbox.grid(row=0, column=i, **padding)

        self.model_label_frame.pack(**padding)

        self.dataset_label_frame = ttk.LabelFrame(self, text='Датасет:')

        self.selected_dataset = tk.StringVar(value=DATASETS[0])
        for i, dataset in enumerate(DATASETS):
            radio_button = ttk.Radiobutton(
                self.dataset_label_frame,
                text=dataset,
                value=dataset,
                variable=self.selected_dataset
            )
            radio_button.grid(row=0, column=i, **padding)

        self.dataset_label_frame.pack(**padding)

        self.slider_frame = ttk.Frame(self)

        self.train_label = ttk.Label(self.slider_frame, text='Использование тренировочной выборки:')
        self.train_label.grid(row=0, column=0, **padding)

        self.train_percentage = tk.IntVar(value=100)
        self.train_slider = ttk.Scale(self.slider_frame, from_=1, to=100, variable=self.train_percentage,
                                      command=self.update_slider_labels)
        self.train_slider.grid(row=0, column=1, **padding)

        self.train_percentage_label = ttk.Label(self.slider_frame)
        self.train_percentage_label.grid(row=0, column=2, **padding)

        self.test_label = ttk.Label(self.slider_frame, text='Использование тестовой выборки:')
        self.test_label.grid(row=1, column=0, **padding)

        self.test_percentage = tk.IntVar(value=100)
        self.test_slider = ttk.Scale(self.slider_frame, from_=1, to=100, variable=self.test_percentage,
                                     command=self.update_slider_labels)
        self.test_slider.grid(row=1, column=1, **padding)

        self.test_percentage_label = ttk.Label(self.slider_frame)
        self.test_percentage_label.grid(row=1, column=2, **padding)

        self.update_slider_labels()

        self.slider_frame.pack(**padding)

        self.num_input_frame = ttk.Frame(self)

        self.epoch_count_label = ttk.Label(self.num_input_frame, text='Число эпох обучения:')
        self.epoch_count_label.grid(row=0, column=0, **padding)

        self.epoch_count = tk.IntVar(value=10)
        self.epoch_count_input = ttk.Entry(self.num_input_frame, textvariable=self.epoch_count)
        self.epoch_count_input.grid(row=0, column=1, **padding)

        self.batch_size_label = ttk.Label(self.num_input_frame, text='Batch size:')
        self.batch_size_label.grid(row=1, column=0, **padding)

        self.batch_size = tk.IntVar(value=4)
        self.batch_size_input = ttk.Entry(self.num_input_frame, textvariable=self.batch_size)
        self.batch_size_input.grid(row=1, column=1, **padding)

        self.num_input_frame.pack(**padding)

        self.begin_button = ttk.Button(self, text='Начать тестирование', padding=7, command=self.start_training)
        self.begin_button.pack(**padding)

        self.pack(padx=10, pady=10)

    def update_slider_labels(self, event=None):
        """Обновление значений ярлыков с процентами."""

        self.train_percentage_label.config(text='{}%'.format(self.train_percentage.get()))
        self.test_percentage_label.config(text='{}%'.format(self.test_percentage.get()))

    def start_training(self):
        """Начало обучения и тестирования моделей."""

        # получение списка выбранных моделей
        selected_models = [model.get() for model in self.selected_models]
        if True not in selected_models:
            showerror(title='Ошибка', message='Не выбрано ни одной модели')
            return
        model_tuples = [self.model_tuples[i] for i in range(len(self.model_tuples)) if selected_models[i]]

        # получение числа эпох обучения
        try:
            epoch_count = self.epoch_count.get()
            if epoch_count < 1:
                raise tk.TclError()
        except tk.TclError:
            showerror(title='Ошибка', message='Ошибка в вводе числа эпох обучения')
            return

        # получение размера батча
        try:
            batch_size = self.batch_size.get()
            if batch_size < 1:
                raise tk.TclError()
        except tk.TclError:
            showerror(title='Ошибка', message='Ошибка в вводе batch size')
            return

        train_and_evaluate(self.selected_dataset.get(), self.train_percentage.get(), self.test_percentage.get(),
                           epoch_count, batch_size, model_tuples)


class TrainingOutputFrame(ttk.Frame):
    pass


class ResultFrame(ttk.Frame):
    pass
