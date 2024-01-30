Данная сеть принимает фото животного (кошка/собака и определяет принадлежность животного к классу).
Кошка -макс. значение на нейроне под индексом 4, собака-под индексом 2.


Данная нейросеть является полно связной и  написана без фреймворков, сторонних библиотек по созданию и обучению неройнных сетей.

Она является универсальной и принимает значения в виде одномерного массива ( сделано для удобства предоставления информации для различных задач).
Так же она может быть переделана в сверточную нейронную сеть. ( будет показано ниже)
Начинаем создания класса NeuraNetwork, который принимает количество сигналов на входном, скрытом и выходном слоем коэффициент обучения(input_nodes, hidden_nodes, output_nodes,
                 lerninggrate).
Создаем случайные матрицы весов между входным и скрытым слое, скрытым и выходным.
создаем коэффициент обучения и функцию активации.

создаем метод  quary, который принимает список входных значений.
далее идет расчет входящих и исходящих сигналов каждого слоя.
#расчет входящих сигналов для скрытого слоя рассчитывается путем умножения матрицы весов ( скрытого и входящего слоя на значения сигнала, поступившего на нейрон).
#расчет исходящего  сигналов для скрытого вычисляется путем применения к полученному входящему сигналу функции активации ( в качестве функции активации выбрана сигмоида).
#для остальных слоев по аналогии.
Метод  quary возвращает выходной сигнал каждого нейрона.

Создаем метод def train,  который принимает список входных значений и список классификации(inputs_list, target_list) объектов( маркеры).
Значения наши объектов переводим в двумерный массив и транспонировкам( для удобства матричного умножения).
Как и методе quary создаем входные/выходные значения для нейронов каждого слоя.
В качество обучения используется метод обратного распространения ошибка путем градиентного спуска от выходного слоя, через скрытый к входному.
Ошибку выходного слоя считаем как разницу между желаемым и полученным значением.
Обучение проходит от выходного слоя к входному и вычисляется как коэффициент обучения * (ошибка выходного слоя * выходной слой * (1-выхдной слой)), массив скрытого слоя тампонированный)
Следующему слоя по аналогии, с учетом поправки на слои.
Задается количество нейронов на каждом слое, задается коэффициент обучения и создается экземпляр класса нейронной сети.

Модуль prepair_data_set - подготовка данных.
Загружаем картинку и изображение кота/ собаки ( закомментировал кусок кода, когда можно через цикл загрузить бесконечно большое количество картинок, что сформирует отличный дата сет и обучит нейросеть).
Картинка представлена в формате GRB и содержит данные в трехмерном массиве.
Переводим данные в одномерный массив  т.к мы изначально написали прием данных в виде одномерного массива (подаем одно значение на один нейрон).
Уменьшаем картинку до размера 28х28 ( для удобства обучения нейросети, т.к нейросеть полно связная)
Изображение 28х28 дает нам 784 нейрона на входящем слое, на которые мы будем подавать по одному значению
Далее преобразовываем данные так, что бы они стали он 0 до 1 ( для нормализации обучения нейронки, если данные будут большими ( в нашем случает от 0 до 255), то обучение может застопориться, т.к градиент может стремиться к 
0, это особенности функции активации сигмоида)
После подготовки данных можем подавать их на неросеть, сохранять обученные веса в csv. файл и в дальнейшем использовать для классификации объектов.

Модуль cnn.
В данном модуле реализован алгоритм свертки для нейронной сети.
После получения данных в виде двумерного массива (30х30) на матрицу накладывается так называемый фильтр в нашем случае представляет собой матрицу 3х3 ( фильтры созданы внутри модуля).
Алгоритм реализуется путем наложения нашего фильтра на матрицу (начинаем с левого угла матрицы, далее сдвигаемся на 1 столбец вправе, доходим до конца матрицы и переходим на 1 строку вниз и т.д)
После наложения фильтра ( свертки нашей матрицы) ее размерность уменьшается с 30х30 до 28х28 ( как раз то, что нам нужно)
Далее подаем эти данные на нашу ранее полученную нейронную сеть и на каждом слое добавляем применение данного фильтра и все! ( фильтры на каждом слое можно использовать разные).











