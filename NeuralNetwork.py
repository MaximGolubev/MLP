from PIL import Image
import numpy as np


class Neuron:
    def __init__(self, inputs=np.array([]), weights=np.array([]), activation_func=lambda x: max(0, x)):
        self.inputs = inputs
        self.activation_func = activation_func
        self.weights = weights
        # self.output = activation_func(np.sum(inputs * weights))

    def get_output(self, inputs):
        print (inputs.shape, self.weights.shape)
        return self.activation_func(np.sum(inputs * self.weights))

# class Layer:
#     def __init__(self, neurons=np.array([]), prev_layer=None, next_layer=None):
#         self.neurons = neurons
#         self.prev_layer = prev_layer
#         self.next_layer = next_layer




class Network:
    def __init__(self,
                 input_num,             # Количество входов
                 hidden_layers_num,     # Количество скрытых слоев
                 neurons_num,           # Количество нейронов в скрытом слое
                 output_num,            # Количество выходов
                 learning_step,         # Шаг обучения
                 error                  # Допустимая ошибка
                 ):

        self.error = error
        self.learning_step = learning_step
        self.neurons_num = neurons_num
        self.output_num = output_num
        self.input_num = input_num

        layers_list = [[] for _ in range(hidden_layers_num + 2)]
        layers_list[0] = [Neuron(weights=np.array([1 for _ in range(input_num)])) for _ in range(input_num)]
        layers_list[1:hidden_layers_num+2] = [
            [Neuron(weights=np.random.uniform(-1.0, 1.0, size=len(layers_list[i-1]))) for _ in range(neurons_num)]
            for i in range(1, hidden_layers_num + 1)
        ]
        layers_list[hidden_layers_num + 1] = [Neuron(weights=np.random.uniform(-1.0, 1.0, size=neurons_num)) for _ in range(output_num)]

        self.layers_list = layers_list

    def feedforward(self, data):
        inputs = np.copy(data)
        for layer in self.layers_list:
            print("NEW LAYER")
            inputs = np.array([neuron.get_output(inputs) for neuron in layer])
        return inputs

if __name__ == "__main__":
    import os
    import struct


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

    print(len(X_train[1]))

    network = Network(X_train.shape[1], 15, X_train.shape[1], 10, 1, 1)
    print(network.feedforward(X_train[1]))