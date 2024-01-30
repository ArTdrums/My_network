import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import scipy.misc

# можно через цикл сразу обработать все картинки, но мы сделаем на примере одной картинки
'''for i in range(2, 5, 2):
    img = Image.open(f'C:\\Users\\Артем\\PycharmProjects\\net_3\\venv\\new\\кошка_{i}.jpeg').resize((28, 28)).convert('L')
    arr = np.asfarray(img, dtype='uint8')'''

img = Image.open('C:\\Users\\Артем\\PycharmProjects\\net_3\\venv\\new\\кошка 3.jpeg').resize((30, 30)).convert('L')
arr = np.asfarray(img, dtype='uint8')

flattened_arr = arr.flatten()

# преобразование данных
scaled_input = (np.asfarray(flattened_arr[:])) / 255.0 * 0.99 + 0.01
print(scaled_input)
# print(scaled_input)
test_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ]  # список маркеров
onodes = 10
# преобразуем список маркеров

"создаю условия, при котором у определеного нейрона будет максимальный коэффециент "
targets = np.zeros(onodes) + 0.01
targets[test_list[2]] = 0.99

for i, item in enumerate(targets):
    print(f'номер нейрона {i} значение нейрона {item}')

plt.imshow(arr, interpolation=None)
plt.show()

# print('длина массива ')
# print(len(flattened_arr))
