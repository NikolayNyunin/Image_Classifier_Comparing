from inspect import getmembers, isfunction

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import models
from train import train_and_evaluate

DATASETS = ('MNIST', 'Fashion-MNIST', 'CIFAR-10')


class App(tk.Tk):
    """Основной класс приложения."""

    def __init__(self):
        super().__init__()

        self.title('Сравнение классификаторов изображений')
        self.resizable(False, False)

        # получение информации о моделях, описанных в файле `models.py`
        self.model_tuples = getmembers(models, isfunction)

        self.container = ttk.Frame(self)
        self.container.pack(fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.parameter_selection_frame = ParameterSelectionFrame(self.container, self)
        self.parameter_selection_frame.grid(row=0, column=0, sticky='nsew')
        self.geometry(self.parameter_selection_frame.geometry)

        self.result_frame = None

    def show_parameter_selection_frame(self):
        self.parameter_selection_frame.tkraise()
        self.geometry(self.parameter_selection_frame.geometry)

    def train_and_show_results(self, dataset_name: str, train_percentage: int, test_percentage: int,
                               epoch_count: int, batch_size: int, model_tuples: list[tuple]):

        metrics, train_times, frame = train_and_evaluate(
            dataset_name,
            train_percentage,
            test_percentage,
            epoch_count,
            batch_size,
            model_tuples
        )

        model_names = [name for name, _ in model_tuples]

        if self.result_frame:
            self.result_frame.destroy()
        self.result_frame = ResultFrame(self.container, self, model_names, metrics, train_times, frame)
        self.result_frame.grid(row=0, column=0, sticky='nsew')
        self.geometry(self.result_frame.geometry)


class ParameterSelectionFrame(ttk.Frame):
    """Класс, описывающий виджет выбора параметров для тестирования моделей."""

    geometry = '480x360'

    def __init__(self, master, controller: App):
        super().__init__(master)

        self.controller = controller

        model_names = [name for name, _ in self.controller.model_tuples]

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
        model_tuples = [self.controller.model_tuples[i]
                        for i in range(len(self.controller.model_tuples)) if selected_models[i]]

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

        self.controller.train_and_show_results(
            self.selected_dataset.get(),
            self.train_percentage.get(),
            self.test_percentage.get(),
            epoch_count,
            batch_size,
            model_tuples
        )


class TrainingOutputFrame(ttk.Frame):
    pass


class ResultFrame(ttk.Frame):
    """Класс, описывающий виджет для отображения результатов тестирования моделей."""

    geometry = '1280x900'

    def __init__(self, master, controller: App, model_names, metrics, train_times, figure):
        super().__init__(master)

        padding = {'padx': 20, 'pady': 5}

        self.table_frame = ttk.Frame(self)

        self.model_name_label = ttk.Label(self.table_frame, text='Название модели', font='Calibri 14')
        self.model_name_label.grid(row=0, column=0, **padding)

        self.accuracy_label = ttk.Label(self.table_frame, text='Итоговая точность', font='Calibri 14')
        self.accuracy_label.grid(row=0, column=1, **padding)

        self.loss_label = ttk.Label(self.table_frame, text='Итоговое значение функции потерь', font='Calibri 14')
        self.loss_label.grid(row=0, column=2, **padding)

        self.train_time_label = ttk.Label(self.table_frame, text='Время обучения (секунд)', font='Calibri 14')
        self.train_time_label.grid(row=0, column=3, **padding)

        for i in range(len(model_names)):
            model_name_label = ttk.Label(self.table_frame, text=model_names[i], font='Consolas 14')
            model_name_label.grid(row=i + 1, column=0, **padding)

            accuracy_label = ttk.Label(self.table_frame, text=round(metrics[i][1], 3), font='Consolas 14')
            accuracy_label.grid(row=i + 1, column=1, **padding)

            loss_label = ttk.Label(self.table_frame, text=round(metrics[i][0], 3), font='Consolas 14')
            loss_label.grid(row=i + 1, column=2, **padding)

            train_time_label = ttk.Label(self.table_frame, text=round(train_times[i], 3), font='Consolas 14')
            train_time_label.grid(row=i + 1, column=3, **padding)

        self.table_frame.pack(expand=True, **padding)

        self.figure_canvas = FigureCanvasTkAgg(figure, self)
        NavigationToolbar2Tk(self.figure_canvas, self)
        self.figure_canvas.get_tk_widget().pack(expand=True, **padding)

        self.return_button = ttk.Button(self, text='Заново', padding=10,
                                        command=controller.show_parameter_selection_frame)
        self.return_button.pack(expand=True, **padding)
