import random
import matplotlib.pyplot as plp
import numpy as np
import scipy.special
from tqdm import tqdm
from PIL import Image
import collections
import csv

from cnn import res_net


img = Image.open('C:\\Users\\Артем\\PycharmProjects\\net_3\\venv\\new\\кошка 3.jpeg').resize((28, 28)).convert('L')
arr = np.asfarray(img, dtype='uint8')
# arr_res = sum(arr, [])
flattened_arr = arr.flatten()
with open('weight_1_1', 'r', encoding='utf-8', newline='') as File:
    reader = csv.reader(File)
    for row in reader:
        row = np.array(list(map(float, row)))
        W_hidden_input_res = row
W_hidden_input_res = np.reshape(W_hidden_input_res, (100, 784))

with open('weight_2', 'r', encoding='utf-8', newline='') as File:
    reader = csv.reader(File)
    for row in reader:
        row = np.array(list(map(float, row)))
        W_output_hidden_res = row
W_output_hidden_res = np.array(W_output_hidden_res)
# res_new = res_new.ravel()

W_output_hidden_res = np.reshape(W_output_hidden_res, (-1, 100))


class NeuraNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 lerninggrate):  # задаем колличество входных, скрытых и выходных сигналов
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # создаем матрицу весов  между входным и скрытыми слоями размерностью скрытый Х входной
        # self.W_hidden_input = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.W_hidden_input = W_hidden_input_res - 0.01

        # далее создаем матрицу весов между скрытый и выходным  размерностью выходной Х скрытый
        # self.W_output_hidden = np.random.rand(self.onodes, self.hnodes) - 0.5 # после обучения сохраняем веса и искользуем сохраненные веса ,
        # подгруженные из файла
        self.W_output_hidden = W_output_hidden_res - 0.01

        # коэффициент обучения
        self.lr = lerninggrate
        # использщуем сигмоиду в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        '''def sigm(x):
            return 1 / (1 +np.exp(-x))'''

        pass

    def train(self, inputs_list, target_list):  # тренировка (уточнение весовых коэффициентов
        #  преобразовываем список входных сигналов в значения двумерного массива
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # расчет входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.W_hidden_input, inputs)  # произведение матрицы на сигнал
        # рассчет исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(
            hidden_inputs)  # для скрытого слоя мы применяем функцию активации к входному слою и получаем матрицу
        # исходящих сигналов скрытого слоя

        # рассчет входящих сигналов для выходного слоя, берем последнею матрицу и умножаем на исходящий сигнал со скрытого слоя
        final_inputs = np.dot(self.W_output_hidden, hidden_outputs)

        # рассчет исходящих  сигналов для выходного слоя, применяя функцию активации.
        final_outputs = self.activation_function(final_inputs)
        # ошибка выходного слоя ( целевой значение - фактическое)
        outputs_error = targets - final_outputs

        hidden_error = np.dot(self.W_output_hidden.T, outputs_error)
        # обновление весов между узлами скрытого слоя и выходного сигнала
        self.W_output_hidden += self.lr * np.dot((outputs_error * final_outputs * (1.0 - final_outputs)),
                                                 np.transpose(hidden_outputs))

        self.W_hidden_input += self.lr * np.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)),
                                                np.transpose(inputs))

        pass

    def quary(self, inputs_list):  # опрос ( получение значений с выходных узлов после
        # предоставление значений входного сигнала
        # преобразовываем список значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        # расчет входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.W_hidden_input, inputs)  # произведение матрицы на сигнал

        # рассчет исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(
            hidden_inputs)  # для скрытого слоя мы применяем функцию активации к входному слою и получаем матрицу
        # исходящих сигналов скрытого слоя

        # рассчет входящи сигналов для выходного слоя, берем последнию матрицу и умножаем на исходящий сигнал со скрытого слоя
        final_inputs = np.dot(self.W_output_hidden, hidden_outputs)

        # рассчет исходящих  сигналов для выходного слоя, применяя функцию активации.
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# задаем колличество слоев
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
final_hidden = 3
# коэфф. обучения
learning_rate = 0.3

onodes = 10
targets = np.zeros(onodes) + 0.01

# создаем экземпляр нейронной сети

n = NeuraNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# создаем матрицу весов между входным и скрытыми слоями размерностью скрытый Х входной
W_hidden_input = np.random.rand(hidden_nodes, input_nodes) - 0.5
# далее создаем матрицу весов  между выходный и скрытым размерностью выходной Х скрытый

W_output_hidden = np.random.rand(output_nodes, hidden_nodes) - 0.5
# print(f'матрица входного и скрытого слоя {W_output_hidden} \n матрица выходного и скрытого слоя {W_output_hidden}')
res = [random.random() - 0.5 for i in range(input_nodes)]

# print(n.quary(flattened_arr[))
epochs = 100
for e in tqdm(range(epochs), desc='процесс обучения'):
    # go through all records in the training data set
    scaled_input = (np.asfarray(flattened_arr[:])) / 255.0 * 0.99 + 0.01

    # print(scaled_input)
    test_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]  # список маркеров

    onodes = 10
    # преобразуем список маркеров
    targets = np.zeros(onodes) + 0.01
    targets[test_list[4]] = 0.99
    n.train(scaled_input, targets)
    pass
print(n.quary(scaled_input))


W_hidden_input = W_hidden_input.flatten()
W_output_hidden = W_output_hidden.flatten()

# запись весов между первый и скрытым слоев  нейросети в файл
with open('weight_1_1', 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(W_hidden_input)

# проверка весов после чтения фала
with open('weight_1_3', 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(W_hidden_input_res)
