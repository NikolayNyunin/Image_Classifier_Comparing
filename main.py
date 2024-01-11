from inspect import getmembers, isfunction

from gui import App, ParameterSelectionFrame
import models


def main():
    # получение информации о моделях, описанных в файле `models.py`
    model_tuples = getmembers(models, isfunction)

    # создание и запуск экземпляра приложения
    app = App()
    ParameterSelectionFrame(app, model_tuples)
    app.mainloop()


if __name__ == '__main__':
    main()
