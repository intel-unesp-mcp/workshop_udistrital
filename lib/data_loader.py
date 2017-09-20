import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

def load_mnist_data_for_cnn():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train =  x_train.reshape((x_train.shape[0],) + input_shape)
    x_test = x_test.reshape((x_test.shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_test /= 255
   
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    
    return ((x_train, y_train), (x_test, y_test))


def load_mnist_data_for_mlp():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    

    # x_train /= 255
    # x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return ((x_train, y_train), (x_test, y_test))


def plot_digit(digit_data, label):
    digit = digit_data.reshape(28,28)
    plt.title('Label is {}'.format(label))
    plt.imshow(digit, cmap='gray')
    plt.show()
