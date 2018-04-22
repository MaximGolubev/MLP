import numpy as np


def relu(x):
    return float(max(0, x))


class Layer(object):
    def __init__(self, weights: np.matrix, biases) -> object:
        self.weights = np.copy(weights)
        self.biases = np.copy(biases)
        self.values = np.zeros((1, weights.shape[1]))

        #print("VAL SHAPE ", self.values.shape, " WEIGHTS SHAPE ", self.weights.shape)

    def calc_values(self, inputs: np.matrix):
        print(inputs[0][0:5], "INPUTS ", self.weights[0])
        print(inputs.shape, self.weights.shape, self.biases.shape)
        self.values = np.matrix(np.vectorize(relu)(np.dot(inputs, self.weights) - self.biases), dtype=np.float32)
        print(self.values[0][0:10], "VALUES", self.values.shape)
        return self.values

    def set_values(self, values):
        self.values = values


class InputLayer(Layer):
    pass


class HiddenLayer(Layer):
    pass


class OutputLayer(Layer):
    pass



class Network(object):
    def __init__(self,
                 input_num,  # Количество входов
                 hidden_layers_num,  # Количество скрытых слоев
                 neurons_num,  # Количество нейронов в скрытом слое
                 output_num,  # Количество выходов
                 learning_step,  # Шаг обучения
                 error  # Допустимая ошибка
                 ):
        self.error = error
        self.learning_step = learning_step
        self.neurons_num = neurons_num
        self.output_num = output_num
        self.input_num = input_num

        layers = [None for _ in range(0, hidden_layers_num + 3)]

        layers[0] = Layer(weights=np.eye(input_num, input_num), biases=np.zeros(shape=(1, input_num)))
        for i in range(1, hidden_layers_num + 2):
            layers[i] = Layer(weights=np.matrix(
                np.random.uniform(low=-1.0, high=1.0, size=(layers[i - 1].values.shape[1], neurons_num))),
                              biases=np.zeros(shape=(1, neurons_num))) #np.random.uniform(low=-1.0, high=1.0, size=(1, neurons_num)))
        layers[hidden_layers_num + 2] = Layer(
            weights=np.matrix(np.random.uniform(low=-1.0, high=1.0, size=(layers[1].values.shape[1], output_num))),
            biases=np.zeros(shape=(1, output_num)))#np.random.uniform(low=-1.0, high=1.0, size=(1, output_num)))

        self.layers = layers

    def feed_forward(self, data):
        data = data.reshape((1, len(data)))
        self.layers[0].set_values(data)
        for layer in self.layers[1:]:
            data = layer.calc_values(data)
        return data

    def fit(self, x, y):
        x_train = np.copy(x)
        y_train = np.copy(y)

        for inputs in x_train:
            output = self.feed_forward(inputs)
            sig




if __name__ == "__main__":
    import os
    import struct

    def one_hot(mas: np.array):
        y = np.zeros(shape=(mas.shape[1], 10))
        for i in range(mas.shape[1]):
            y[i][mas[i]] = 1.0
        return y

    def load_mnist(path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                   '%s-labels.idx1-ubyte'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images.idx3-ubyte'
                                   % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                     lbpath.read(8))
            labels = np.fromfile(lbpath,
                                 dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII",
                                                   imgpath.read(16))
            images = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(labels), 784)

        return images, labels


    X_train, y_train = load_mnist('E:/Perceptron', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

    X_test, y_test = load_mnist('E:/Perceptron', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255  # Normalise data to [0, 1] range
    X_test /= 255  # Normalise data to [0, 1] range

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    network = Network(X_train.shape[1], 5, 50, 10, 1, 1)
    print(network.feed_forward(X_train[0]), " ", y_train[0])
