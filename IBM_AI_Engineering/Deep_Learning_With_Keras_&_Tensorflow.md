# Глубокое обучение на Keras и Tensorflow

## Содержание
[Расширенные функциональные возможности Keras](#title1)   
[   - введение в расширенный Keras  ](#title2)     
[   - функциональный API Keras и API подклассификации  ](#title3)       
[   - создание пользовательских слоев Keras  ](#title4)  
[   - обзор TensorFlow 2.x   ](#title5)  
[Усовершенствованные CNN в Keras](#title6)  
[   - усовершенствованные CNN в Keras   ](#title7)  
[   - методы расширения данных   ](#title8)  
[   - трансферное обучение   ](#title9)  


## <a id="title1">▶️ Расширенные функциональные возможности Keras </a>  
### <a id="title2"> Введение в расширенный Keras </a>
Keras широко используется в промышленности и науке для разных приложений: от распознования изображений и речи, до систем рекомендаций и обработки естественного языка

>[!NOTE]
>Благодаря своей простоте Keras является фаворитом среди специалистов по глубокому обучению. Google, Netflix, Uber и др используют данную библиотеку для разработки решений с ИИ

Keras содержжит
- последовательный API - подходит для простого линейного стека слоев
~~~Python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
  Dense(64, activation="relu", input_shape=(784,)),
  Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
~~~
- функциональный API - обеспечивает создание более сложных моделей благодаря большй гибкости и контролю
~~~Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
~~~

#### Функциональный API
Позволяет создать более сложные модели, чем линейный стек (несколько входов/выходов, общие слои, непоследовательные потоки данных)

Порядок построения функционального API:
1) Определяем входной слой
2) Определяем уровни и выходной слой
3) Объединяем входной и выходной слой в модель

Преимущества функционального API:
1) Гибкость - позволяет создавать сложные архитектуры (прим модели с несколькими ветвями, остаточные соединения)
2) Более понятная и легко отслеживаемая структура модели
3) Слои и модели можно повторно использовать в разных частях архитектуры





### <a id="title3"> Функциональный API Keras и API подклассификации </a>
Основной синтаксис:
~~~Python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


#Объявляем входной слой
input_layer = Input(shape=(20,))
print(input_layer)

#Добавляем скрытые слои
'''64 нейрона в слое, каждый скрытый слой принимает на вход выход предыдущего слоя'''
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(64, activatio='relu')(hidden_layer1)

#Выходной слой
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

#Создаем модель указывая входные и выходные слои
'''Соединяются входной и выходные слои ерез скрытые слои'''
model = Model(input=input_layer, output=output_layer)
'''Краткое изложение  модели'''
model.summary()

#Компиляция модели
'''(оптимизатор, функция потерь, оценка точности во время обучения)'''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Тренировка модели
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

#Оценка модели на тестовых данных
X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))

'''.evaluate вычисляет потери и точность модели на тестовых данных'''
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
~~~

Одним из преимуществ функционального API является возможность обработки моделей с несколькими входами и выходами. Это особенно полено в сложных приложениях таких как многозадачное обучение:
~~~Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

#Определяем вход
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

#Определяем плотный слой
x = Dense(8, activation='relu')(inputA)
x = Dense(4, activation='relu')(x)
x = Model(inputs=inputA, outputs=x)

y = Dense(16, activation='relu')(inputB)
y = Dense(4, activation='relu')(y)
y = Model(inputs=inputB, outputs=y)

#Объединяем выходы 2 ветвей
combined = concatenate([x.output, y.output])

z = Dense(2, activation='relu')(combined)
z = Dense(1, activation='linear')(z)

#Создаем модель
model = Model(inputs=[x.input, y.input], outputs=z)
~~~

#### API подклассов
*API подклассов* - Еще одна мощная функция Keras. В отличие от последовательных и функциональных API обеспечивается наибольшая гибкось. Данный API позволяет определить собственные и динамические модели, путем создания подкласса модели и реализации собственного метода call()

API подклассов полезны когда:
- нужно создать модели в которых прямой переход не может быть определен статически и нужно динамически изменять архитектуру
- когда нужно лучше контролировать обучение и хотим внедрить собственные циклы обучения
- исследования и создание прототипов для экспериментов с новыми типами слоев или архитектур недоступных в стандартно Keras API

Чтобы использовать API подклассов нужно создать подкласс модели и определить слои в метода init. Затем реадизовать *forvard pass* в методе call() где явно определяем как слои связаны и данные проодят через модель
~~~Python
import tensorflow as tf
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

  def call(self, inputs):
    x=self.dense1(inputs)
    return self.dense2(x)
~~~


#### Отсев и пакетная нормализация
*Отсев (Dropout)* - метод регуляризации помогающий предотвратить переобучение в нейросетях. Во время обучения *Dropout* случайным образом устанавливает долю входных единиц на ноль при каждом цикле обновления, это предотвращает черезмерную зависимость модели от любых конкретных нейронов, что побуждает нейросеть изучать более надежные функции, которые лучше обобщают невидимые данные. Отсев применяется только во время тренировки

Слой Отсева:
~~~Python
dropout_layer = Dropout(rate=0.5)(hidden_layer)
~~~

*Пакетная нормализация (Batch normalization)* - метод применяющийся для улучшения тренировочной стабильности и скорости нейросетей. Нормализует выход предыдущего слоя путем центрирования и масштабирования данных, что стабилизирует процесс обучения

Слой батчинга:
~~~Python
from tensorflow.keras.layers import BatchNormalization, Dense, Input

batch_norm_layer = BatchNormalization()(hidden_layer)
~~~


### <a id="title4"> Создание пользовательских слоев Keras </a>
Настраиваемые слои позволяют определять собственные операции в нейросетях, что адаптирует модели к конкретным задачам и эксперементировать с новыми идеями

Создание пользовательских слоев необходимо когда нужны определенные функции, не предоставляемые Keras (Внимание(Q,V,K), Позиция слов в тексте, Борьба с дисбалансом), раскрывая весь потенциал библиотеки

> [!NOTE]
> Стандартные слои Keras:
> Dense - просто умножение матриц
> Conv2D - свертка для картинок

>[!IMPORTANT]
> В 90% случаях достаточно готовых слоев Keras, но при реализации трансформеров и сложных архитектур без пользовательских слоев не обойтись

Настраиваемые уровни:
- помогают внедрить новые методы или алгоритмы непосредственно в модели
- оптимизируют производительность
- адаптируют слои к работе с конкретными данными или вычислительными ограничениями
- повышают гибкость
- позволяют определить уникальное поведение
- улучшает читаемость и обслуживание инкапсулируя сложную логику и повторно используемые компоненты

Базовая структура пользовательского слоя:
1) __init__ - конструктор, сохраняет гиперпараметры
2) build(input_shape) - вызывается один раз когда слой впервыые получает данные, создает веса (add_weight). input_shape показывает какой формы должен быть вес
3) call(inputs) - вся математика, вызывается при каждом прохождении данных

Простой настраиваемый слой выполняющий операцию с высокой плотностью
~~~Python
from tensorflow.keras.layers
import Layer
import tensorflow as tf

class MyCustomLayer(Layer):

  #Метод инициализирующий атрибуты слоя
  def __init__(self, units=32, **kwargs):
    super(MyCustomLayer, self).__init__(**kwargs)
    self.units = units

  #Метод создающий веса слоя
  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(input_shape[-1],
         self.units),
         initializer='random_normal',
         trainable=True
    )
    self.b = self.add_weight(
        shape=(self.units,),
        initializer='zeros',
        trainable=True
    )

  def call(self, inputs):
    #Выполнение плотной операции с последующей активацией реле
    #tf.matmul - матричное умножение
    return tf.matmul(inputs, self.w) + self.b

print(tf.executing_eagerly())
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
result=tf.add(a,b)
print(result)


#Исползованиие своего слоя в модели
from tensorflow.keras.models import Sequential
model = Sequential([
  MyCustomLayer(64),
  MyCustomLayer(10)
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
~~~

#### Практика: создание пользовательского слоя Keras
~~~Python
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout


#Определяем пользовательский слой
class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, inputs):
        '''Функция активации ReLU'''
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# Создаем модель Keras (Многоклассовая классификация)
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),
    Dense(10, activation='softmax')
])


#Компилируем модель
model.compile(optimizer='adam', loss='categorical_crossentropy',)
print("Model summary before building:")
model.summary()

model.build((1000,20))
print("\nModel summary after building:")
model.summary()


#Тренируем модель
'''Генерируем рандомные данные'''
x_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
model.fit(x_train, y_train, epochs=10, batch_size=32)


#Оценка модели
x_test = np.random.random((200, 20))
y_test = np.random.randint(10, size=(200, 1))

y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

loss = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')


#Визуализируем архитектуру модели
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


#Добавляем слой Dropout для предотвращения переобучения
'''Регулизация-случчайно выключает 50% нейронов'''
model = Sequential([
    CustomDenseLayer(64),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=10, batch_size=32)
~~~


### <a id="title5"> Обзор TensorFlow 2.x </a>
**TensorFlow** - платформа с открытым исходным кодом для машинного обучения от Google. 

Ключевые функции TensorFlow:
1) Жадное выполнение (Eager execution) - делает TensorFlow более понятным за счет немедленного выполнения операций (немедленная обратная связь) без построения графиков, упрощая процесс отладки и облегчая интерактивное понимание
2) Интуитивное высокоуровнево API - интегрирут Keras, упрощая процесс создания и обучения моделей глубокого обучения. Keras предоставляет удобный интерфейс позволяющий абстрагироваться от многих сложностей с программированием нейронных сетей, позволяя создавать модели простым и лаконичны кодом
3) Поддержка разных платформ и устройств - поддерживает развертывание на мобильных, веб и эмулированных устройствах что обеспечивает масштаируемость, универсальность и производительность
4) Плавное масштабирование между CPU, GPU, TPU - оптимизирует производиельность для крупномассштабных задач ML
5) Богатая экосистема - широкий спектр инструментов и библиотек, расширяющих возможности TensorFlow:
- TensorFlow Lite (развертывание моделей на мобильных уустройствах)
- TensorFlow JS (развертывание в веб и среде JS)
- TFX Tensorflow (развертывание конвейеров ML)
- TesnorBoard (набор инструментов визуализации, дающий представление о процессе обучения моделей, метрики и графики)


## <a id="title6">▶️ Усовершенствованные CNN в Keras </a>
### <a id="title7"> Усовершенствованные CNN в Keras </a>
**CNN** - нейроети предназначенные для анализа визуальных данных, имитируя зрительную систему человека. Состоят из: сверточных слоев, слоев объединения и полносвязных слоев

Базовая модель CNN Keras:
~~~Python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequentil([
  Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
  MaxPooling2D((2,2)),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
~~~

Более современные архитектуры могут значительно повысить производительность при выполнении сложных задач:
1) VGG - архитектура известная простотой и глубиной. Состоит из сверточных слоев с небольшими фильтрами 3х3, за которым следуют слои с максимальным количеством пулов и полностью соединенных слоев
~~~Python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequentil([
  Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Conv2D(128, (3,3), activation='relu'),
  Conv2D(128, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Conv2D(256, (3,3), activation='relu'),
  Conv2D(256, (3,3), activation='relu'),
  MaxPooling2D((2,2)),
  Flatten(),
  Dense(512, activation='relu'),
  Dense(512, activation='relu'),
  Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
~~~

2) ResNet - архитектура использующая остаточные соединения помогающие обучить глубокие сети, решая проблему исчезающего градиента. Остаточные соединения помогают сети изучать сопоставление идентификаторов
~~~Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

'''Функция остаточного блока'''
def residual_block(x, filters, kernel_size=3, stride=1):
  shortcut = x
  x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
  x = BatchNormalization()(x)
  x = Add()([x, shortcut])
  x = Activation('relu')(x)
  return x

input = Input(shape=(64,64,3))
x = Conv2D(64, (7,7), strides=1, padding='same')(input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = residual_block(x, 64)
x = residual_block(x, 64)
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(input=input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
~~~
3) inception
В них представлены такие концепции как:
- более глубокие сети
- остаточные соединения
- многомасштабное извлечение функций

### <a id="title8"> Методы увеличения данных (аугументация) </a>
**Аугументация** - искусственное создание новых данных из уже имеющихся. Борется с переобучением и улучшает обобщение

В Keras для аугументации на лету используют класс ImageDataGenerator

>[!IMPORTANT]
>Для классификации изображений (определения что на картинке - один объект на все изображение) лучше подходит Keras/VGG16, для детекции (какаие объекты на изображении) и обвода рамками лучше использовать YOLO

Методы увеличения данных:
1) Ротация
2) Переворачивание
3) Масштабирование
4) Добавление шума

>[!NOTE]
>Методы аугументации применяются только к тренировочным данным и путем добавления разнообразия позволяет избежать переобучения модели

Для увеличения данных CNN в Keras используется ImageDataGenerator
~~~Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
import numpy as np

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Загружаем и подготавливаем данные
image_paths = [
    'img/training_images1.jpg',
    'img/training_images2.jpg',
    'img/training_images3.jpg'
]

training_images = []
for image_path in image_paths:
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    training_images.append(img_array)
training_images = np.array(training_images)


i = 0
for batch in datagen.flow(training_images, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    plt.title(f'Augmented Image {i + 1}')
    i += 1
    if i % 4 == 0:
        break
plt.show()
~~~

Keras позволяет использовать продвинутые методы:
- функциональная нормализация
- нормализация на основе выборки
- применение настраиваемых функций дополнения

Пример функциональной и выборочной нормализации:
~~~Python
datagen = ImageDateGenerator(
  '''Задаем среднее набора данных = 0'''
  featurewise_center=True,
  '''Нормализуем набор данных и имеем стандартное отклонение = 1'''
  featurewise_std_normaliztion=True,
  '''Задаем среднее значение каждой выборки = 0'''
  samplewise_center=True,
  '''Нормализуем стандартное отклонение каждой выборки = 1'''
  samplewise_std_normalization=True
)

#Вычисляем среднее значение и стандартное отклонение набора данных изображений
datagen.fit(training_images)

#datagen.flow генерирует нормализованные изображения из обучающего набора
i = 0
for batch in datagen.flow(training_images, batch_size=32):
  plt.figure(i)
  imgplot = plt.imshow(image.array_to_img(batch[0]))
  i+=1
  if i % 4 ==0
    break
plt.show()
~~~

Пример: настраиваемая функция увеличения которая добавляет случайный шум к изображениям
~~~Python
import numpy as np

def add_random_noise(image):
  noise = np.random.normal(0, 0.1, image_shape)
  return image + noise

#Определяем настраиваемую функцию для изображения
datagen = ImageDataGenerato(processing_function = add_random_noise)
~~~

### <a id="title9"> Трансферное обучение </a>
Трансферное обучение использует уже полуенные знания путем повторного использования моделей, подготовленных на больших и полных наборах данных. Концепция отражает как люди используют старые знания для решения новых, но связанных задач

Трансферное обучение включает в себя использование предварительно обуенной модели обученной на большом наборе данных (таком как ImageNet) и доработку ее на новом наборе

Используя VGG16 предоубученную на миллионах изображений и умеющую распознавать края, текстуры, формы и другие общие характеристики, мы можем использовать как сверточную базовую модель для наших задач экономя тем самым время и вычислительные ресурсы, также уменьшая размер необходимых для обучения данных
