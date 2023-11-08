from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical


def train_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Get values from 0 to 1
    x_train = x_train / 255
    x_test = x_test / 255

    # Pre-processing
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Sequential model
    model = Sequential([
        # Flatten layer - input of the network, 32x32 image with 3 channels
        Flatten(input_shape=(32, 32, 3)),
        # Dense layer - hidden layer with 1000 neurons
        Dense(1000, activation='relu'),
        # Dense Layer - output layer with 10 neurons (10 diferent results). Softmax - gives the probability for every possible category and all sum up to 1
        Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
    model.save('saved_models/cifar10_model.h5')


if __name__ == "__main__":
    train_model()