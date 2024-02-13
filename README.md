Для начала пронумеруем светофоры для присвоения им id.
Составляем мат модель для решения задачи, для этого нам потребуется:
# ответить на вопрос чем мы будем управлять? -фазами светофора
# получение данных (фиксация трафика) на каждый светофор.
# разборка алгоритма включения светофором в зависимости от трафика
# оптимизация с помощью нейронной сети

Поиск оптимального решения в работе светофором будет достигаться путем анализа входных значения на каждый светофор. При показателе около 20ед на один или несколько светофором нейронная сеть будет принимать решение об включении/отключении светофоров (выход -> 0 выключение, выход  -> 1 включение)


Получение входных данных для светофоров.
Т.к данные взять неоткуда по условию задачи, я сгенерировал их самостоятельно (по своему усмотрению).
каждый светофор принимает на себя определенное значение трафика(от 0 до бесконечности), я взял от 0 до 35. 
!!!!на светофор всегда приходят какие-то значения!!!!
Светофоры включаются при определенном количестве трафика, поступившего на них, данные всегда фиксируются на всех 12 светофорах, неважно сколько там 0 или 35.

Алгоритм включения светофоров.
Я выделил несколько основных движений:
# по вертикали сверху вниз (1, 2, 5, 6, 7, 8)
# по горизонтали снизу-вверх (3, 4, 10, 9, 11, 12)
# поворот направо снизу-вверх (вертикальное движение) (1, 10, 9, 7, 8)
# поворот на права сверху вниз (вертикальное движение) (2, 5, 6, 11, 12)
# поворот направо (горизонтальное движение слева направо) (4, 6, 5, 9,10)
# поворот направо (горизонтальное движение справа налево) (3, 7, 8, 11,12)
# для пешеходов (5, 6, 7, 8, 9, 10, 11, 12)
Желтый цвет не учитывал т.к он не запрещает и не разрешает движение, а для нашей тестовой сети это усложнит реализацию.
Исходя из данной логики был создан дата фрейм для реализации логики, который включает по 4 обучающих входящих выборки для каждого маневра

Нейронная сеть написана 2 способами (оба способа идентичные, но не много с разной архитектурой)
1. С помощью фреймворка tensorflow.
2. Рукописная нейросеть.

Коротко о работе нейросети
В качестве архитектуры выбрана полно связная нейронна сеть, состоящая из:
# 12 нейронов на входящем слое
# 6 неронов на скрытом_1 слое
# 3 нейронов на скрытом_2 слое
# 1 нероне выходного слоя
Функция активация выбрана линейная, обучение(Backpropagation) 'стохастический градиентный спуск', функция потерь 'mse', метрика 'mae'
На выходном слое по каждому светофору (id -0 до 11) показывается вероятность (от 0 до 1) включения данного светофора.
Для проверки работы неросети нужно подать 12 значений в диапазоне от (0-35 ) в виде одномерного массива и получите на выходе ответ неросети.

Пример:
test_net = [6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]
for i, item in enumerate(model.predict(np.array(test_net) / 25.0 * 0.99 + 0.01)):
    print(светофор под индексом {i}, вероятность включения {item}')



Рукописная нейронная сеть.
В качестве архитектуры выбрана полно связная нейронна сеть, состоящая из:
# 12 нейронов на входящем слое
# 24 неронов на скрытом_1 слое
# 12 выходного слоя
Данная сеть принимает одномерный массив данных (1 цисло на один световор) и выдает вероятность включения светофора (индекс- вероятность).
Выход сети - на каждом нейроне показывается своя вероятность включения каждого светофора ( 1 нейрон на выходе - один светофор со свои индексом)



Данная нейросеть является полно связной и написана без фреймворков, сторонних библиотек по созданию и обучению неройнных сетей.
Начинаем создания класса NeuraNetwork, который принимает количество сигналов на входном, скрытом и выходном слоем коэффициент обучения (input_nodes, hidden_nodes, output_nodes,
                 lerninggrate).
Создаем случайные матрицы весов между входным и скрытым слое, скрытым и выходным.
создаем коэффициент обучения и функцию активации.
создаем метод quary, который принимает список входных значений.
далее идет расчет входящих и исходящих сигналов каждого слоя.
#расчет входящих сигналов для скрытого слоя рассчитывается путем умножения матрицы весов (скрытого и входящего слоя на значения сигнала, поступившего на нейрон).
#расчет исходящего сигналов для скрытого вычисляется путем применения к полученному входящему сигналу функции активации (в качестве функции активации выбрана сигмоида).
#для остальных слоев по аналогии.
Метод quary возвращает выходной сигнал каждого нейрона.

Создаем метод def train, который принимает список входных значений и список классификации (inputs_list, target_list) объектов(маркеры).
Значения переводим в двумерный массив и транспонируем (для удобства матричного умножения).
Как и методе quary создаем входные/выходные значения для нейронов каждого слоя.
В качество обучения используется метод обратного распространения ошибка путем градиентного спуска от выходного слоя, через скрытый к входному.
Ошибку выходного слоя считаем, как разницу между желаемым и полученным значением.
Обучение проходит от выходного слоя к входному и вычисляется как коэффициент обучения * (ошибка выходного слоя * выходной слой * (1-выхдной слой)), массив скрытого слоя тампонированный)
Следующему слоя по аналогии, с учетом поправки на слои.
Задается количество нейронов на каждом слое, задается коэффициент обучения и создается экземпляр класса нейронной сети.
Переходим к обучению, использую написанный ранее df.
для проверки корректности работы неройсети подаем одномерный массив из данных пример:
test_net_1 = np.array([6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]) / 25.0 * 0.99 + 0.01
# проверка нейросети
for i, item in enumerate(n.quary(test_net_1)):
    print(светофор под индексом {i}, вероятность включения {item}')
