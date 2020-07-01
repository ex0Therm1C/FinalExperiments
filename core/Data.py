import numpy as np
import tensorflow.keras as keras

def labelToOneHot(x, numLabels=10):
    result = np.zeros((len(x), numLabels), dtype=np.float32)
    # result[np.arange(len(x)), x[:,0]] = 1
    result[np.arange(len(x)), x] = 1
    return result

def loadMNIST(numTest=1000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    x_train = np.array(x_train, dtype=float) / 255
    y_train = labelToOneHot(y_train, numLabels=10)

    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    y_test = labelToOneHot(y_test[:numTest], numLabels=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def loadMNISTLocal(numTest=1000):
    x_train = np.load('../mnist/x_train.npy').reshape(-1, 28, 28, 1)
    x_train = np.array(x_train, dtype=float) / 255
    y_train = labelToOneHot(np.load('../mnist/y_train.npy'), numLabels=10)
    x_test = np.load('../mnist/x_test.npy').reshape(-1, 28, 28, 1)[:numTest]
    x_test = np.array(x_test, dtype=float) / 255
    y_test = labelToOneHot(np.load('../mnist/y_test.npy')[:numTest], numLabels=10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def loadCifar(numTest=1000, numLabels=10):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, numLabels)
    x_train = np.array(x_train, dtype=float) / 255
    y_test = keras.utils.to_categorical(y_test[:numTest], numLabels)
    x_test = np.array(x_test, dtype=float)[:numTest] / 255
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test
