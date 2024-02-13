from keras.layers import Dense  # это класс полно связной нейросети
import tensorflow as tf
from keras.models import Sequential  # это класс последовательно слоев - можно создавать
import numpy as np
import pandas as pd

from keras.activations import linear, sigmoid
from df_svetofor import *

# уходим от случайной генерации весов

tf.random.set_seed(7)
model = Sequential([
    Dense(12, input_shape=(1,), activation='linear'),
    Dense(6, input_shape=(1,), activation='linear'),
    Dense(3, input_shape=(1,), activation='linear'),
    Dense(1, input_shape=(1,), activation='linear')

])

model.compile(optimizer='sgd', loss='mse', metrics='mae')

# запускаем обучение для вертикального движения
for i in df_vertikal:
    print(model.fit(i, y_vertikal, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_vertikal),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для горизонтального движения

for i in df_gorixont:
    print(model.fit(i, y_gorixont, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_gorixont),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для поворота на право снизу вверх

for i in df_povorot_1_ssnizy_vverh_vertikal:
    print(model.fit(i, y_povorot_1_snizy_vverh_vertikal, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_povorot_1_snizy_vverh_vertikal),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для поворота на право сверху вниз


for i in df_povorot_1_sverhy_vnis_vertikal:
    print(model.fit(i, y_povorot_1_sverhy_vnis_vertikal, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_povorot_1_sverhy_vnis_vertikal),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для отдельного включения пешеходных переходов

for i in df_pesh:
    print(model.fit(i, y_pesh, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_pesh),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для поворота на право при горизонтальном движении справа налево

for i in df_povorot_1_sverhy_vnis_gorizont:
    print(model.fit(i, y_povorot_1_sverhy_vnis_gorizont, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_povorot_1_sverhy_vnis_gorizont),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# запускаем обучение для поворота на право при горизонтальном движении справа налево
for i in df_X_povorot_2_sverhy_vnis_gorizont:
    print(model.fit(i, y_povorot_2_sverhy_vnis_gorizont, epochs=100))
    print(pd.DataFrame({
        'верное значение': np.squeeze(y_povorot_2_sverhy_vnis_gorizont),
        'пердсказания': np.squeeze(model.predict(i)),
    }))

# тестирование нейронной сети

''' округление доцелого числа если надо
b = np.array([int(i) for i in np.squeeze(model.predict(X))])

print('было ', w1, w0, '\n', 'стало', w2, w3)
'''
# тестирование нейронной сети
test_net = [6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]
for i, item in enumerate(model.predict(np.array(test_net) / 25.0 * 0.99 + 0.01)):
    print(f'светофор под индексом {i}, вероятность включения {item}')


