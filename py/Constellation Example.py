from constellation import *
from random import randint

# создание объекта типа Constellation, инициализация параметрами группировки Starlink из конфига
constellation = Constellation('constellationsTest.json', 'Starlink')

# вычисление элементов орбиты для всех КА в начальный момент
constellation.get_initial_state()

# определение точек на оси времени, в которые будут проихзводиться расчёты
epochs = list(range(1002))

# расчёт положений всех КА в заданные моменты времени
constellation.predict(epochs)

# Координаты случайного КА (в инерциальных осях) после этого можно прочитать из constellation.state_eci
sat_idx = randint(0, constellation.total_sat_count - 1)
epoch_idx = randint(0, len(epochs) - 1)
print("Положение КА-" + str(sat_idx) + " на эпоху " + str(epochs[epoch_idx]) + ":")
print(constellation.state_eci[sat_idx, :, epoch_idx])

# Поиск оптимального значения поля зрения с заданной точностью
answer = 180. / np.pi * constellation.search_fov_threshold(3, 0)
if answer > 0:
    print('Оптимальное значение поля зрения КА для обеспечения глобальности покрытия: ', end='')
    print(f'{answer} градусов')
else: # код -1
    print('Глобальное покрытие невозможно для заданной конфигурации')

# Визуализация
constellation.visualize_coverage(answer, 0)
