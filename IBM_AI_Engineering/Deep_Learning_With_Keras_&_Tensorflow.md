# Глубокое обучение на Keras и Tensorflow

## Содержание


## Расширенные функциональные возможности Keras
### Введение в расширенный Keras
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





### Функциональный API Keras и API подклассификации
Основной синтаксис:
~~~Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

#Определяем вход
inputs = Input(shape=(784,))

#Определяем плотный слой
x = Dense(64, activation='relu')(inputs)

#Определяем выходной слой
outputs = Dense(10, activation='softmax')(x)

#Создаем модель
model = Model(inputs=inputs, outputs=outputs)
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
