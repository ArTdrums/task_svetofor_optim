import numpy as np

# входные значения для вертикального движения
X_vertikal_1 = np.array([25, 20, 4, 1, 18, 19, 18, 19, 2, 2, 2, 2]) / 25.0 * 0.99 + 0.01
X_vertikal_2 = np.array([25, 18, 5, 2, 19, 20, 19, 20, 3, 3, 3, 3]) / 25.0 * 0.99 + 0.01
X_vertikal_3 = np.array([25, 21, 6, 3, 20, 20, 19, 20, 4, 3, 4, 3]) / 25.0 * 0.99 + 0.01
X_vertikal_4 = np.array([25, 19, 2, 4, 20, 21, 20, 17, 1, 2, 3, 4]) / 25.0 * 0.99 + 0.01
# формируем обучающую выборку для входные значения для вертикального движения
df_vertikal = [X_vertikal_1, X_vertikal_2, X_vertikal_3, X_vertikal_4]
# выходные значения вертикального движения
y_vertikal = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

# входные  значения для горизонтального движения
X_gorixont_1 = np.array([7, 8, 7, 35, 30, 4, 3, 4, 23, 23, 25, 23]) / 35.0 * 0.99 + 0.01
X_gorixont_2 = np.array([6, 6, 8, 34, 29, 2, 4, 2, 22, 22, 23, 26]) / 35.0 * 0.99 + 0.01
X_gorixont_3 = np.array([5, 5, 6, 33, 28, 3, 2, 3, 25, 20, 22, 20]) / 35.0 * 0.99 + 0.01
X_gorixont_4 = np.array([7, 8, 7, 35, 30, 4, 3, 4, 23, 23, 25, 23]) / 35.0 * 0.99 + 0.01

# формируем обучающую выборку входных для горизонтального движения
df_gorixont = [X_gorixont_1, X_gorixont_2, X_gorixont_3, X_gorixont_4]

# выходные значения для горизонтального движения
y_gorixont = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])

# входные значения для поворота направо при вертикальном движении вверх
X_povorot_1_snizy_vverh_vertikal = np.array([25, 3, 3, 2, 1, 3, 22, 22, 21, 21, 3, 3]) / 25.0 * 0.99 + 0.01
X_povorot_2_snizy_vverh_vertikal = np.array([22, 5, 5, 3, 4, 3, 25, 21, 18, 21, 1, 3]) / 25.0 * 0.99 + 0.01
X_povorot_3_snizy_vverh_vertikal = np.array([21, 2, 3, 5, 3, 4, 19, 25, 23, 25, 2, 6]) / 25.0 * 0.99 + 0.01
X_povorot_4_snizy_vverh_vertikal = np.array([18, 2, 3, 2, 1, 3, 18, 17, 21, 24, 1, 1]) / 25.0 * 0.99 + 0.01

# формируем обучающую выборку для поворота направо при вертикальном движении вверх
df_povorot_1_ssnizy_vverh_vertikal = [X_povorot_1_snizy_vverh_vertikal, X_povorot_2_snizy_vverh_vertikal,
                                      X_povorot_3_snizy_vverh_vertikal,
                                      X_povorot_4_snizy_vverh_vertikal]
# выходные значения для поворота на права снизу вверх
y_povorot_1_snizy_vverh_vertikal = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

# входные значения для поворота на права при вертикальном движении сверху вниз
X_povorot_1_sverhy_vnis_vertikal = np.array([2, 22, 2, 2, 18, 21, 2, 2, 2, 2, 21, 18]) / 25.0 * 0.99 + 0.01
X_povorot_2_sverhy_vnis_vertikal = np.array([25, 3, 3, 2, 1, 3, 22, 22, 21, 21, 3, 3]) / 25.0 * 0.99 + 0.01
X_povorot_3_sverhy_vnis_vertikal = np.array([25, 3, 3, 2, 1, 3, 22, 22, 21, 21, 3, 3]) / 25.0 * 0.99 + 0.01
X_povorot_4_sverhy_vnis_vertikal = np.array([25, 3, 3, 2, 1, 3, 22, 22, 21, 21, 3, 3]) / 25.0 * 0.99 + 0.01

df_povorot_1_sverhy_vnis_vertikal = [X_povorot_1_sverhy_vnis_vertikal, X_povorot_2_sverhy_vnis_vertikal,
                                     X_povorot_3_sverhy_vnis_vertikal,
                                     X_povorot_4_sverhy_vnis_vertikal]
# выходные значения для поворота на права при вертикальном движении сверху вниз
y_povorot_1_sverhy_vnis_vertikal = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1])

# входные значения для поворота на права при горизонтальном движении слева направо

X_povorot_1_sverhy_vnis_gorizont_1 = np.array([3, 4, 2, 25, 22, 22, 2, 5, 22, 21, 2, 2]) / 25.0 * 0.99 + 0.01
X_povorot_1_sverhy_vnis_gorizont_2 = np.array([1, 3, 2, 18, 25, 18, 3, 2, 21, 24, 5, 6]) / 25.0 * 0.99 + 0.01
X_povorot_1_sverhy_vnis_gorizont_3 = np.array([5, 1, 4, 22, 17, 21, 5, 1, 22, 22, 2, 1]) / 25.0 * 0.99 + 0.01
X_povorot_1_sverhy_vnis_gorizont_4 = np.array([5, 2, 6, 25, 22, 25, 2, 5, 25, 21, 3, 4]) / 25.0 * 0.99 + 0.01

df_povorot_1_sverhy_vnis_gorizont = [X_povorot_1_sverhy_vnis_gorizont_1, X_povorot_1_sverhy_vnis_gorizont_2,
                                     X_povorot_1_sverhy_vnis_gorizont_3, X_povorot_1_sverhy_vnis_gorizont_4]
# выходные значения для поворота направо при горизонтальном движении с лева направо

y_povorot_1_sverhy_vnis_gorizont = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])

# входные значения для поворота направо при горизонтальном движении справа налево

X_povorot_2_sverhy_vnis_gorizont_1 = np.array([3, 4, 22, 2, 3, 3, 22, 25, 2, 2, 22, 22]) / 25.0 * 0.99 + 0.01
X_povorot_2_sverhy_vnis_gorizont_2 = np.array([1, 3, 23, 4, 5, 5, 21, 22, 3, 4, 25, 21]) / 25.0 * 0.99 + 0.01
X_povorot_2_sverhy_vnis_gorizont_3 = np.array([5, 5, 25, 5, 6, 6, 25, 21, 4, 5, 20, 25]) / 25.0 * 0.99 + 0.01
X_povorot_2_sverhy_vnis_gorizont_4 = np.array([6, 6, 25, 6, 4, 4, 25, 22, 3, 4, 25, 25]) / 25.0 * 0.99 + 0.01

df_X_povorot_2_sverhy_vnis_gorizont =[X_povorot_2_sverhy_vnis_gorizont_1, X_povorot_2_sverhy_vnis_gorizont_2, X_povorot_2_sverhy_vnis_gorizont_3, X_povorot_2_sverhy_vnis_gorizont_4]
# выходные значения для поворота направо при горизонтальном движении справа налево
y_povorot_2_sverhy_vnis_gorizont = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])



# дата фрейм отдельное включение пешеходных переходов
X_pesh_1 = np.array([5, 6, 7, 8, 18, 21, 17, 19, 19, 22, 21, 18]) / 22.0 * 0.99 + 0.01
X_pesh_2 = np.array([6, 5, 8, 7, 22, 18, 22, 21, 17, 19, 19, 22]) / 22.0 * 0.99 + 0.01
X_pesh_3 = np.array([3, 8, 5, 4, 17, 16, 19, 22, 22, 17, 18, 15]) / 22.0 * 0.99 + 0.01
X_pesh_4 = np.array([3, 4, 2, 5, 22, 17, 12, 16, 19, 15, 22, 22]) / 22.0 * 0.99 + 0.01

# формируем обучающую выборку для пешеходных переходов
df_pesh = [X_pesh_1, X_pesh_2, X_pesh_3, X_pesh_4]
# выходные значения для отдельное включение пешеходных переходов
y_pesh = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
