from tensorflow import keras


def a_basic_ann() -> keras.Sequential:
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def b_basic_cnn() -> keras.Sequential:
    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def c_advanced_cnn() -> keras.Sequential:
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(28, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
