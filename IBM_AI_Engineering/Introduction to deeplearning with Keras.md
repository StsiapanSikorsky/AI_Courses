# Введение в глубокое обучение и нейронные сети с Keras

В основе глубокого обучения лежат нейронные сети

Примеры применения CNN и RNN:
1) Восстановление цвета на изображении в оттенках серого
2) Синтез аудиоклипа с видео и синхронизация губ со звуками и словами
3) Автоматическая генерация рукописного текста
4) Автоматический перевод текста на изображении на лету
5) Добавление звука к немым фильмам
6) Классификация объектов на изображении
7) Преобразование текста в изображение
8) Чат боты

Все алгоритмы в глубоком обучении основаны на понимании как нейронные сети функционируют и обрабатывают данные в мозге. Основная часть нейрона называется *сомой* и содержит ядро нейрона. Сеть нейрона называется *дендридами*. Плечо в другом направлении называется *аксоном*. Усы на конце апсона называются *синапсами*

Дендроны получают электрические импульсы передающие информацию (данные) от соседних нейронов. Далее данные передаются в сому. В ядре данные обрабатываются и передаются аксону, затем аксон передает информацию на синапс и выход нейрона становится входом для тысяч других. Обучение в мозге происходит путем многократной активации одних нейронных связей по сравнению с другими.


## Искусственные нейронные сети
Искусственный нейрон ведет себя также, как биологический. Он состоит из сомы, дендритов и аксона. Конец аксона может разветвляться и соединяться со многими другими нейронами.

**Пресептрон** - официальное название искусственного нейрона. Он вычисляет взвешенную сумму входных данных и сравнивает ее с пороговым значением, если сумма превышает порог он выдает "1", если нет "0". Это фундаментальный строительный блок искусственных нейросетей

В нейросети сеть нескольких искусственных нейронов обединяются в слои. Нейросеть обычно разделяется на разные слои:
1) Входной слой - передает входные данные в сеть
2) Скрытые слои - любой набор узлов между входными и выходными слоями
3) Выходной слой

### Прямое распространение
**Прямое распространение** - метод при помощи которого данные передаются через слои нейрона в нейросети от входного слоя к выходному.

Каждое соединение в нейроне имеет определенный вес, при помощи которого регулируется поток данных. После применения весов к данным нейроны обрабатывают эту информацию, выводя взвешенную сумму входных данных, к сумме также может добавляться константа (смещение). Взвешенная сумма сопоставляется нелинейным пространством. Популярной функцией является *Сигмоидная функция*. Данная функция (не обязательно сигмоида) называется *ФУНКЦИЕЙ АКТИВАЦИИ*

Функция активации - выполняет нелинейное преобразование входных данных и решает следует ли активировать нейрон (актуальная ли информация или ее следует игнорировать). Нейросеть без функции активации по сути просто модель линейной регрессии

![2.1](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.1.png)

Практика: расчет выхода модели по коэффициентам

![2.2](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.2.png)
~~~Python
import numpy as np

#Инициализируем веса
weights = np.around(np.random.uniform(size=6), decimals=2)
#Инициализируем смещения
biases = np.around(np.random.uniform(size=3), decimals=2)
print(weights)
print(biases)

#Инициализируем входы
x_1 = 0.5
x_2 = 0.85
print('x1 is {} and x2 is {}'.format(x_1, x_2))

#Вычисляем взвешенные суммы (согласно схемы)
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_12))

#При помощи сигмоиды вычисляем функцию активации первого и второго узла
#Это выходы двух прецептронов скрытого слоя являются входами для выходного слоя
a_11 = 1.0 / (1.0 + np.exp(-z_11))
a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

#Выходной слой
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))
a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))
~~~

Практика: Вычислние для любого количества входов, выходов и скрытых слоев
~~~Python
#Определяем структуру сети (ее конфигурация)
n = 2 '''количество входов'''
num_hidden_layers = 2 '''количество скрытых слоев'''
m = [2, 2] '''по 2 нейрона в каждом слое'''
num_nodes_output = 1 '''количество выходов'''


#Счетчик, сколько нейронов в перыдущем слое
num_nodes_previous = n

#Созаем пустой словарь (для хранения всей сети)
network = {}

#Цикл создающий структуру нейросети
'''У каждой нейросети есть столько весов, сколько нейронов в предыдущем слое'''
for layer in range(num_hidden_layers + 1):
    #Определяем имя слою и количество нейронов
    #Для последнего слоя специальное имя, остальные  нумерованные
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]

    #Создаем нейроны для данного слоя
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    num_nodes_previous = num_nodes

#Выводим нейросеть
print(network)

#Инициализируем веса и смещения для сети с любым количеством скрытых слоев и количеством узлов в каждом слое
#Функция инициализации
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    '''+1 так как нужно сформировать не тоько скрытые слои, а и выходной'''
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes
    return network


#Создаем сеть (5 входов, 3 скрытых слоя)
#(1 слой - 3 узла, 2 - 2, 3 - 3)
#(1 - выходной слой)
small_network = initialize_network(5, 3, [3, 2, 3], 1)

#Функция вычисления взвешенной суммы на каждом узле
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

#Входные данные
np.random.seed(12)
'''Генерирует 5 случайных чисел от 0 до 1'''
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('Входные данные в нейросеть {}'.format(inputs))

#Берем нейрон в 1 узле 1 скрытого слоя (одного нейрона)
node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

#Считаем взвешенную сумму
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('Взвешенная сума 1 узла 1 скрытого слоя = {}'.format(np.around(weighted_sum[0], decimals=4)))

#Определяем функцию сигмоиды
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

#Вычисляем вывод 1 узла первого скрытого слоя
node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
print('Вывод 1 узла 1 скрытого слоя: {}'.format(np.around(node_output[0], decimals=4)))


#РАЗОБРАТЬСЯ
def forward_propagate(network, inputs):
    '''Входы'''
    layer_inputs = list(inputs)

    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        #Выходы текущего слоя становятся входаами следующего
        layer_inputs = layer_outputs

    network_predictions = layer_outputs
    return network_predictions

#Вычисление прогнозирования сети
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))

my_network = initialize_network(5, 3, [2, 3, 2], 3)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))
~~~

## Основы глубокого обучения
### Градиентный спуск (Gradient Descent)

Проблматика: Найти такое значение весов, чтобы модель лучше всего предсказывала правильные ответы:
z = w · x где:  
x — входные данные (знаем)  
z — правильные ответы (знаем)  
w — вес, который нужно найти (не знаем)  

Понять что вес w хороший мы можем через функцию фильтрации потерь (парабулу) (чем меньше значение, тем лучше вес w:  
**J(w)=∑(z—w⋅x)^2**

Задача сводится к нахождению минимума этой параболы - это и есть лучшее w. В решении данной задачи и помогает градиентный спуск

**Градиентный спуск** - итеративный алгоритм оптимизации для нахождения минимума функции, используется для нахождения наилучшего значения веса w. Лучший алгоритм когда нужно оптимизировать много весов. Чтобы найти минимум функции при помощи градиентного спуска мы делаем шаги, пропорциональные отрицательному значению градиента функции в текущей точке

**Градиент** - наклон функции потерь

Формула обновления веса в градиентном спуске:   
![2.5](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.5.png)   

где: 
w_new - новое значение веса (куда придем после шага)  
w_old - старое значение веса (где сейчас находимся)  
а - размер шага  
dJ - производная функции потерь  
dw - производная по весу  
dJ/dw - градиент (наклон функции потерь в текущей точке)  


Работа алгоритма:  
1) Выбираем случайное начальное значение w0  
2) Начинаем делать шаги к максимуму функции (минимуму параболы) где w = 2 (зеленая точка) по формуле обновления веса градиента  
4) Делаем шаг и переходим к w1 (синяя точка)  
5) Повторяем алгоритм пока не достигнем минимального значения (значения функции стоимости) близкого к минимуму в пределах порогового значения  

![2.3](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.3.png)

![2.4](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.4.png)


> [!IMPORTANT]
> Нужно быть осторожным при выборе скорости (α):  
> высокая скорость обучения приведет к большим шагам которые могут привести к упущению минимального балла  
> небольшая скорость обучения может привести к очень маленьким шагам и большому времени для нахождения минимальной точки  




### Backpropagation (алгоритм вычисления градиентов для всех весов модели)

Проблема: нейросеть имеет миллион весов, после прямого распространения (forward propagation) мы получаем ошибку. Нужно понять какой вес виноват в ошибке.

**Backpropagation** - алгоритм вычисления градиентов распространяющий ошибку назад по сети и говорящий каждому весу на сколько он неверный. Считает dJ/dw для каждого веса

Процесс:
1) Данные проходят по нейросети
2) На выходе вычисление ошибки "Е" между прогнозируемым значением и основными метками истины. Теперь ошибка представляет собой функцию потерь (стоимость)
3) Распространение "Е" обратно в нейросеть и выполнение градиентного спуска на разных весах и смещениях в сети для оптимизации весов w. Для каждого веса корректируется на сколько он не правильно внес вклад
4) Обновляем веса и смещения путем обратного распространения
5) Повторяем шаги пока не достигнем максимально заданного количества итераций или пока прогнозируемый результат не станет ниже порогового значения

Математическое обоснование:
Ошибка (Е) зависит от выхода (а) который зависит от веса (z)

Пример применени в PyTorch:
~~~Python
import torch

# Forward
output = model(X)
loss = criterion(output, y)

# Backward (backpropagation)
loss.backward()   # ← PyTorch делает всё сам

# Обновление весов
optimizer.step()
optimizer.zero_grad()
~~~

Практика: Применение алгоритма Backpropagation  
~~~Python
import numpy as np
import matplotlib.pyplot as plt

#Определяем вход и выход для XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
d = np.array([0, 1, 1, 0])

#Задаем параметры сети и веса
def initialize_network_parameters():
    #Параметры сети
    inputSize = 2      #(x1, x2)
    hiddenSize = 2     #Количество скрытых слоев
    outputSize = 1     #Количество выходных нейронов
    lr = 0.01           #a-скорость обучения
    epochs = 1000000    #Количество эпох

    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1
    b2 = np.random.rand(outputSize, 1) * 2 - 1

    return w1, b1, w2, b2, lr, epochs

#Обучение нейросети
'''Инициализируем параметры'''
w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

'''Обучаем нейросеть используя Backpropagation'''
error_list = []
for epoch in range(epochs):
    '''Прямое распространение'''
    z1 = np.dot(w1, X) + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    '''Подсчет ошибки'''
    error = d - a2
    da2 = error * (a2 * (1 - a2))
    dz2 = da2 #градинт выходного


    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * (a1 * (1 - a1))

    '''Обновление весов'''
    w2 += lr * np.dot(dz2, a1.T)
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)
    w1 += lr * np.dot(dz1, X.T)
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)

    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

#Тестирование нейросети
z1 = np.dot(w1, X) + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = np.dot(w2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))

print('Final output after training:', a2)
print('Ground truth', d)
print('Error after training:', error)
print('Average error: %0.05f'%np.average(abs(error)))

#Построение графика ошибки
plt.plot(error_list)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
~~~

## Исчезающий градиент (Vanishing gradient)
Из за использования сигмоиды в качестве функции активации, все промежуточные значения находятся в диапазоне от 0 до 1. При обратном распространении мы умножаем множетели меньше единицы друг на друга и как правило их градиенты становятся все меньше. Из за этого нейроны на малых уровнях обучаются очень медленно по сравнению с нейронами последующих уровней. В результате процесс обучения занимает слишком много времени, а точность прогнозов снижается

Соответственно сигмоидные и аналогичные функции не используют в качестве функции активации, поскольку они подвержены проблеме исчезающего градиента

### Функции активации (Activation Functions)
Типы функций активаций для построения нейросетей:
1) Двоичная ступенчатая функция
2) Линейная функция
3) Сигмоидная функция - широко применялась, но из за проблемы исчезающего градиента практически не применяется в многослойных нейросетях. Также имеет недостаток что все ее значения применяют только положительный характер  
![2.6](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.6.png)    
4) Функция гиперболического тангенса (tanh) - похожа на сигмоидную функцию, но симметрична относительно оси начала координат от -1 до 1. Также приводит к проблеме исчезающего градиента в очень глубоких нейросетях  
![2.7](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.7.png)  
5) Выпрямленная линейная единица функция (ReLU) - наиболее широко используемая функция активации. Преимуществом является то что она не активирует все нейроны одновременно. Если входные данные отрицательны они будут преобразованы в 0, а нейрон не активируется. Одновременно активируется лишь несколько нейронов, что делает сеть разряженной и очень эффективной. Данная функция решила проблему исчезающего градиента. Используется только в сырых слоях  
![2.8](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.8.png)  
6) Функция протечки ReLU
7) Softmax функция - разновидность сигмоидной функции, но удобна при решении задач классификации где на выходе нужно получить вероятности принадлежности входного класса

Практика: исследование функций активации: Sigmoid и ReLU и tanh
~~~Python
import numpy as np
import matplotlib.pyplot as plt

#Определяем сигмоидную функцию активации
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Определяем гидрафную сигмоиду
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

#Определяем ReLU
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

#Генерируем 400 входных значений от -10 до 10
z = np.linspace(-10, 10, 400)

#Вычисляем производные функций сигмоиды и ReLU
sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)

#Визуализируем функции и их градиенты
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()

#Определяем функцию тангенса гиперболического
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

#Сравнение функции тангенса с функцией ReLU
z = np.linspace(-5, 5, 100)

tanh_grad = tanh_derivative(z)
relu_grad = relu_derivative(z)

plt.figure(figsize=(12, 6))

# Plot Relu and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()


# Plot tanH and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, tanh(z), label='tanH Activation', color='g')
plt.plot(z, tanh_grad, label="tanH Derivative", color='r', linestyle='--')
plt.title('tanH Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()
~~~

## Keras и библиотеки глубокого обучения
### Библиотеки глубокого обучения
1) TensorFlow - самая популярная библиотека. Чаще всего используется для создания моделей глубокого обучения. Выпущена в 2015, разработана Google
2) PyTorch - основан на платформе Torch и языке Luar. Поддерживает алгоритмы машинного обучения работающие на графических процессорах. Выпущен в 2016. Становится предпочтительнее, чем применение библиотеки TensorFlow в академических исследованиях и приложениях глубокого обучения
3) Keras - высокоуровневый API для создания моделей глубокого обучения. Проще в освоении и применении чем TensorFlow и PyTorch. Keras обычно работает поверх более низкоуровневой библиотеки TensorFlow (для использования Keras в начале нужно устаовить TensorFlow)
4) Theano - одна из первых устаревшихх библиотек глубокого обучения. Больше не поддерживается

>[!NOTE]
>Еслли хотим создать нейростеть быстро можно использовать Keras. Но если необходимо лучше контролировать разные узлы и уровни сети или следить что происходит с сетью с течением времени, необходимо применть PyTorch или TensorFlow


### Регрессионные модели с помощью Keras
Поскольку регрессия представляет собой пресказание непрерывного значчения, на выходе всегда один нейрон БЕЗ ФУНКЦИИ АКТИВАЦИИ (линейный выход)


В библиотеке Keras есть две модели
1) Последовательная модель (*Sequential*)
2) Класс модели используемый в функциональном API

Пример создания простой регрессивной модели с помощью Keras:
~~~Python
import keras
from keras.model import Sequential
from keras.layers import Dense

#Создание модели
model = Sequential()

#Создание слоев (количество нейронов в каждом слое 5, функция активации ReLU
n_cols = concrete_data.shape[1]
model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#adam-Adam в качестве алгоритма оптимизации
#MSE-мера потерь
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors, target)

#Делаем прогнозы
predictions = model.predict(test_data)
~~~

Практика: Использование Keras для построения регрессионной модели
~~~Python
import os

#Подавление предупреждений о использовании CPU
#При использовании модели на GPU просто закоментить
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

import warnings
warnings.simplefilter('ignore', FutureWarning)



#Загружаем и очищаем датасет
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

concrete_data.head()
concrete_data.shape
concrete_data.describe()
concrete_data.isnull().sum()

#Разделение данных на предикторы и целевые
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

predictors.head()
target.head()

#Нормализуем данные
'''Путем вычитания среднего и деления на стандартное отклонение'''
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1]

#Построение нейросети
def regression_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Тренировка и построение сети
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)
~~~

### Классификационные модели с помощью Keras
Процесс создания классификационной модели с Keras сводится к четырем шагам:
1) Создание модели
2) Компиляция
3) Обучение
4) Оценка

При решении задачи классификации с Keras нам необходимо преобразовать целевой столбец в массив с двоичными данными (используя функцию *toCategorical*)

Вотличие от решения задачи регрессии, выходных нейронов может быть столько, сколько классов мы классифицируем, также следует назначать выходным нейронам функцию активации (для решения задач двоичного классификатора будет достаточно применить сигмоиду, а для множественной классификации стоит рассмотреть функцию softmax)

Пример создания классификационной модели с Keras (8 входных предиктов и 4 выходных нейрона классификации:
~~~Python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

model = Sequential()

n_cols = car_data.shape[1]
target = to_categorical(target)

model.add(Dense(5, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
~~~

Практика: обучение классификации Keras на наборе данных MIST
~~~Python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist

#Загружаем данные из библиотеки MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape """(6000, 28, 28) - 6к изображение 28х28 пикселей"""

#Визуализируем первое изображение
plt.imshow(X_train[0])

#Сглаживаем изображение в одномерные векторы (поскольку в обычной нейросети по другому мы не можем подать)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#Нормализуем векторы (поскольку значения пикселей варьируется в пределах 0-255)
X_train = X_train / 255
X_test = X_test / 255

#Разделяем целевую переменную на категории
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

#Определяем функцию посмтроения модели
def classification_model():
    model = Sequential()
    model.add(Input(shape=(num_pixels,)))
    model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Тренируемм и тестируем модель
model = classification_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

#Расчитываем точность модели
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))

#Сохраняем модель (.keras)
model.save('classification_model.keras')

#Используем функцию загрузки модели (когда понадобится еще ее использовать)
pretrained_model = keras.saving.load_model('classification_model.keras')
~~~

## Модели глубокого обучения
### Неглубокие и глубокие нейронные сети
**Мелкая нейронная сеть** - сеть содержащая 1 или 2 скрытых слоя и принимает входные параметры только в виде векторов

Глубокие нейронные сети имеют от 3 и более скрытых слоев и множество нейронов в них, они спообны принимать необработанные данные (прим. изображения или текст) и автоматически извлекают функции для лучшего изучения данных

Резкое увеличение производительности и внедрения нейросетей обусловлено факторам:
1) Прогресс в самой области (например использование ReLU)
2) Доступность данных (глубокие нейросети лучше работают при обучении с большими объемами данных), чем больше данных мы передадим, тем лучше работают нейросети
3) Повышение вычислительных мощностей

### Сверточные нейронные сети CNN
CNN - сверточные нейронные сети предназначенные для решения задач распознования изображений, обнаружения объектов и компьютерного зраения. Они понимают что на вход подаются изображения в формате nxnx1 для серых оттенков и nxnx3 для RGB. CNN имеют значительно меньше параметров за счет использования фильтров (общих весов), что ускоряет обучение и снижает риск переобучения


#### Архитектура CNN:

![2.9](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.9.png)  

Сеть состоит из:
**1) Входной слой**
**2) Сверточные слои (ReLU)**
В сверточном слое мы определяем фильтры и вычисляем свертку между определенными фильтрами и каждым из 3 изображений. Фильтр наносится на изображение и вычисляет произведение между точками и сохраняет значение в пустую ячейку матрицы. Далее шаги повторяются с перемещением фильтра с определенным шагом (stride). Этапы повторяются пока не будет пройдено все изображение и матрица не заполнится. Те же дествия применяются к зеленым и синим изображениям. Сверточный слой также содержит ReLU

![2.10](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.10.png)  

**3) Слои объединения (Pooling Layer)**
Слой объединения - уменьшает пространственные размеры данных распространяемых по сети, уменьшает вычислительную сложность и дает модели пространственную инвариантность (сеть узнает изображение даже если он немного сдвинут на картинке)
Существует два слоя объединения:
*- Максимальное количество пулов (наиболее распространенное)* - для каждого сканируемого участка изображения сохраняется наибольшее значение (например фильтр перемещается по 2 шага за раз)
  
![2.11](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.11.png)  

*- Среднее количество пулов*- вычисляется среднее значение на каждой сканирующей области
  
![2.12](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.12.png)  

**4) Полностью связанный слой**
Входные данные последнего сверточного слоя сглаживаются (Flattern) и соединяем каждый узел текущего слоя с каждым узлом следуующего слоя. Количество нейронов в слое ровняется количеству классов в рассматриваемой задаче (например при классификации цифр на изображении количество нейронов = 10)

#### Применение Keras для построения CNN
~~~Python
model = Sequential()

#Изображения 128х128px
input_shape = (128, 128, 3)
model.add(Input(shape=(input_shape)))

#Сверточный слой с 16 фильтрами 2х2 с шагом 1 в горизонтальном и 1 в вертикальном направлении
model.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), activation='relu'))

#Слой объединения, используем максимальное количество пула с размером фильтра 2х2 с шагами 2,2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#Добавляем еще набор сверточных слоев (используем больше фильтров
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Выравниваем выходные данные сверточных слоев, для перехода их полностью связанные слои
model.add(Flattern())
#Плотый слой со 100 узлами
model.add(Dense(100, activation='relu'))
#Выходной слой равный количеству классов в расматривваемой задаче
model.add(Dense(num_classes, activation='softmx'))
~~~

#### Практика: построене CNN модели на Keras
~~~Python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def convolutional_model():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = convolutional_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
~~~

### Рекуррентные нейронные сети (RNN)
**Рекуррентные нейронные сети (RNN)** - сети с петлями которые не только принимают новые входные данные за раз, но и принимают в качестве входных данных входные данные из преддущей точки данных поступившие в сеть. Данный вид сетей хорошо моделирует закономерности и последовательности данных, там где хронологический порядок имеет значение (текст, геномы, рукописный текст, фондовые рынки). Алгоритмы RNN учитывают время и последовательность

**LSTM (Long Short Term Memory)** - популярная модель улучшенной RNN с долговременной памятью. Используется для генерации изображений, генерации рукописного текста

RNN обрабатывают данные последовательно, что замедляет процесс обучения. Подходят для задач с короткими зависимостями и плохо работают с большим контекстом, а также могут страдать от исчезающих градиентов

На данный момент данный вид нейросетей RNN/LSTM практически вытеснены трансформерами

### Трансформеры (Transformers)
**Трансформеры** - разновидность архитектуры нейросетей. Данная архитектура произвела революцию в обработке естественного языка (NLP). Трансформеры используются для создания моделей глубокого обучения и лежат в основе ChatGPT/Gemini. Трансформеры отлично улавливают долгосрочные зависимости в последоватеьностях данных (в отличие от CNN и RNN) и они идеально подходят для задач машинного перевода, суммирования текста, ответа на ввопроы и генерации текста в изображение. Клюевым в трансформерах является механизм внимания

**Механизм самовнимания** - механизм позволяющий модели оценивать важность разных частей входной последовательности при обработке каждого ввода или токена. Обращая внимание на соответствующие части входных данных трансформеры могут эффективно выявлять зависимости на больших расстояниях (Например в предложении Собака не перешла улицу, потому что она была слишком устала механизм самовнимания поймет что *она* скорее относится к *собака*, чем к *улица*)

#### Трансформеры используют механизм самовнимания для обработки тектовых данных состоящий из частей:
1) Для кадого входного токена генерируются векторы запроса(Q)/ключа(K)/значения(V)
Q-представляет собой слово, на котором мы сосредоточены (Что я ищу?)
K-все остальные слова (Что я могу предложить?)
V-содержит информацию которую нужно передать следующему слою (Что я реально предлогаю?)

2) Оценка внимания - вектор запроса токена Q сравнивается с ключевым вектором К всех других токенов при помощи точечного воспроизведения. В результате получается набор баллов внимания (на сколько они связаны), указывающий на соответствие каждого токена текущему токену

3) Взвешенная сумма - оценки внимания норализуются (через softmax) и для получения векторов контекста вычисляется взвешенная сумма векторов значений для текущего токена

#### Пример:
1) Каждое слово на входе представлено вектором или вложением во внутреннее векторное пространство (сохраняющим позиционную информацию слова в предложении)

![2.13](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.13.png)  

2) Далее мы генерируем Q, K и V из кадого входного вектора

![2.14](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.14.png)  

3) Для каждого слова мы вычисляем как оно должно соответствовать каждому другому слову, значения в матрице ниже можно назвать **баллами внимания**

![2.15](https://github.com/StsiapanSikorsky/AI_Courses/blob/main/IBM_AI_Engineering/img/2.15.png)  

4) Масштабируем оценку и применяем функцию *softmax* для нормализации оценок и превращения их в вероятности

> Softmax("The") = Softmax (2,4,4) --> [0, 0.5, 0.5]

5) Для каждого вектора вычисляем взвешенную сумму векторов значений, взвешенных по оценкам внимания (они используются ля содания нового контекстуального векторного представления

#### Генерация текста в изображение
Для генерации текста в изображение также широко применяются трансформеры, но уже используют механизм *Механизм перекрестного внимания*

**Механизм перекрестного внимания** - позволяет последовательности данных одного типа (текстовая подсказка), влиять на генерацию последовательности данных другого типа (изображения). 

Алгоритм:
1) Механизм изучает контекст встраивания на основе всего предложения
2) Вложения передаются через кодировщик-преобразователь который выдает вектор запросов Q
3) Модель преобразователя изображений (прим DALL-E) использует механизм перекрестного внимания для генерации изображения на основе Q
4) Модель преобразователя изображений генерирует изображение используя вариант авторегрессионной модели, модель предсказывает следующую часть изображения на основе текстовой подсказки и ранее созданных частей изображения

В отличие от RNN, которые обрабатывают последовательность данных, трансформеры можно распараллеливать, что значительно ускоряет повышение квалификации. Это особенно эффективно для машинного перевода, генерации текста и других NLP

Поскольку трансформеры полагаются на обучение на основе денных для определения взаимосвязей, им требуется огромное количество обучающих данных. В этом и заключается главный недостаток трансформров - унаследование подвязанности данных

#### Практика:
~~~Python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from keras.layers import Layer
import warnings
warnings.simplefilter('ignore', FutureWarning)

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import AdditiveAttention, Concatenate, Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

#Подготавливаем данные
input_texts = ["Hello.", "How are you?", "I am learning machine translation.", "What is your name?", "I love programming."]
target_texts = ["Hola.", "¿Cómo estás?", "Estoy aprendiendo traducción automática.", "¿Cuál es tu nombre?", "Me encanta programar."]

target_texts = ["startseq " + x + " endseq" for x in target_texts]

#Преобразуем текст из предложения в токены, используя Tokenizer()
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(target_texts)
output_sequences = output_tokenizer.texts_to_sequences(target_texts)

input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

#?????
max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

#Готовим данные для тренировки
decoder_input_data = output_sequences[:, :-1]
decoder_output_data = output_sequences[:, 1:]

#Инвертируем one-hot
decoder_output_data = np.array([np.eye(output_vocab_size)[seq] for seq in decoder_output_data])


#Слой самовниимания
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is a list: [q_shape, k_shape, v_shape]
        feature_dim = input_shape[0][-1]

        self.Wq = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wq'
        )

        self.Wk = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wk'
        )

        self.Wv = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='Wv'
        )

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Expect list: [query, key, value]
        q, k, v = inputs

        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)

        # Scaled dot-product attention
        scores = K.batch_dot(q, k, axes=[2, 2])
        dk = K.cast(K.shape(k)[-1], dtype=K.floatx())
        scores = scores / K.sqrt(dk)

        attention_weights = K.softmax(scores, axis=-1)
        output = K.batch_dot(attention_weights, v)

        return output

#Архитектура модели
#Кодировщик
encoder_inputs = Input(shape=(max_input_length,))
encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)
encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

#Декодировщик
decoder_inputs = Input(shape=(max_output_length - 1,))
decoder_embedding = Embedding(output_vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

# Attention: decoder attends to encoder outputs
self_attention = SelfAttention()
attention_output = self_attention(
    [decoder_outputs, encoder_outputs, encoder_outputs]
)

# Combine decoder outputs with attention context
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_output])

# Final Dense layer
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat)

# Full Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()


#Обучение модели
history_glorot_adam = model.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

plt.plot(history_glorot_adam.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
~~~



