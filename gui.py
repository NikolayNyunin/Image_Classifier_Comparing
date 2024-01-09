import tkinter as tk
from tkinter import ttk

DATASETS = ('MNIST', 'Fashion-MNIST', 'CIFAR-10')


class ParameterSelectionFrame(tk.Frame):
    """Класс, описывающий окно выбора параметров для тестирования моделей."""

    def __init__(self, master, model_names):
        super().__init__(master)

        options = {'padx': 5, 'pady': 5}

        self.model_label = ttk.Label(self, text='Выберите модели для тестирования:')
        self.model_label.pack(**options)

        self.selected_models = [tk.BooleanVar(value=True) for _ in range(len(model_names))]
        for i, model_name in enumerate(model_names):
            checkbox = ttk.Checkbutton(
                self,
                text=model_name,
                variable=self.selected_models[i],
                onvalue=True,
                offvalue=False
            )
            checkbox.pack(fill='x', **options)

        self.dataset_label = ttk.Label(self, text='Выберите датасет:')
        self.dataset_label.pack(**options)

        self.selected_dataset = tk.StringVar(value=DATASETS[0])
        for dataset in DATASETS:
            radio_button = ttk.Radiobutton(
                self,
                text=dataset,
                value=dataset,
                variable=self.selected_dataset
            )
            radio_button.pack(fill='x', **options)

        self.train_percentage = tk.IntVar(value=100)
        self.train_slider = ttk.Scale(self, from_=1, to=100, variable=self.train_percentage)
        self.train_slider.pack(fill='x', **options)

        self.test_percentage = tk.IntVar(value=100)
        self.test_slider = ttk.Scale(self, from_=1, to=100, variable=self.test_percentage)
        self.test_slider.pack(fill='x', **options)

        self.epoch_label = ttk.Label(self, text='Число эпох обучения:')
        self.epoch_label.pack(**options)

        self.epoch_count = tk.IntVar(value=10)
        self.epoch_input = ttk.Entry(self, textvariable=self.epoch_count)
        self.epoch_input.pack(**options)

        self.batch_size_label = ttk.Label(self, text='Batch size:')
        self.batch_size_label.pack(**options)

        self.batch_size = tk.IntVar(value=4)
        self.batch_size_input = ttk.Entry(self, textvariable=self.batch_size)
        self.batch_size_input.pack(**options)

        self.begin_button = ttk.Button(self, text='Начать тестирование')
        self.begin_button.pack(side='bottom', fill='x', **options)

        self.pack(fill='both', padx=20, pady=20)


class TrainingOutputFrame(tk.Frame):
    pass


class ResultFrame(tk.Frame):
    pass
