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
#Определяем структуру сети
n = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer


#Инициализируем веса и смещения как случайные числа
num_nodes_previous = n  # number of nodes in the previous layer

network = {}  # initialize network an an empty dictionary

# loop through each layer and randomly initialize the weights and biases associated with each node
# notice how we are adding 1 to the number of hidden layers in order to include the output layer
for layer in range(num_hidden_layers + 1):

    # determine name of layer
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]

    # initialize weights and biases associated with each node in the current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }

    num_nodes_previous = num_nodes

print(network)  # print network

#Инициализируем веса и смещения для сети с любым количеством скрытых слоев и количеством узлов в каждом слое
#Функция инициализации
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs  # number of nodes in the previous layer

    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'  # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)  # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]

            # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes
    return network  # return the network

#Создаем сеть (5 входов, 3 скрытых слоя)
#(1 слой - 3 узла, 2 - 2, 3 - 3)
# (1 - выходной слой)
small_network = initialize_network(5, 3, [3, 2, 3], 1)

#Функция вычисления взвешенной суммы на каждом узле
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)
print('The inputs to the network are {}'.format(inputs))

#Вычисляем вщвешенную сумму в 1 узле 1 скрытого слоя
node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']

weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))

#Определяем функцию сигмоиды
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

#Вычисляем вывод 1 узла первого скрытого слоя
node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))


#РАЗОБРАТЬСЯ
def forward_propagate(network, inputs):
    layer_inputs = list(inputs)  # start with the input layer as the input to the first hidden layer

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

        layer_inputs = layer_outputs  # set the output of this layer to be the input to next layer

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
