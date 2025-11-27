"""
Список исправлений:
 - Названия классов, переменных и констант приведены к стандарту PEP8
 - Константы параметров задачи сделаны неизменяемыми на основе NamedTuple
 - Явная инициализация класса WalkerGroup, без наследования от Walker с базовым NamedTuple

Список добавлений:
 - Добавлены комментарии
 - Добавлена обработка возможных ошибок при загрузке данных из JSON
 - Добавлена визуализация группировки с заданным полем зрения

Используемые сокращения:
 - AOL    Argument of Latitude                     аргумент широты
 - RAAN   Right Ascension of the Ascending Node    долгота восходящего узла
 - SMA    Semi-Major Axis                          большая полуось орбиты
"""

import numpy as np
import json
from typing import NamedTuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Constants(NamedTuple):
    EARTH_RADIUS: float   = 6378135           # Экваториальный радиус Земли [m]
    EARTH_GM: float       = 3.986004415e+14   # Гравитационный параметр Земли [m3/s2]
    EARTH_J2: float       = 1.082626e-3       # Вторая зональная гармоника геопотенциала


CONST = Constants()


class Walker(NamedTuple):
    """Описание группировки, константы"""

    inclination: float            # наклонение орбиты
    sats_per_plane: int           # число КА в каждой орбитальной плоскости группы
    plane_count: int              # число орбитальных плоскостей в группе
    f: int                        # фазовый сдвиг по аргументу широты между КА в соседних плоскостях
    altitude: float               # высота орбиты
    max_raan: float               # максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей)
    start_raan: float             # прямое восхождение восходящего узла для первой плоскости


class WalkerGroup:
    """ Данные группировки """

    def __init__(self, inclination: float,
                 sats_per_plane: int,
                 plane_count: int,
                 f: int,
                 altitude: float,
                 max_raan: float,
                 start_raan: float):
        self.walker = Walker(inclination, sats_per_plane, plane_count, f, altitude, max_raan, start_raan)
        if sats_per_plane <= 0 or plane_count <= 0 or abs(inclination) > 90:
            raise ConstellationDataError('Ошибка в данных группировки')

    def get_total_sat_count(self):
        """ Полное число спутников группировки """

        return self.walker.sats_per_plane * self.walker.plane_count

    def get_initial_elements(self):
        """ Расчет начальных значений положения спутников группировки """

        start_raan  = np.deg2rad(self.walker.start_raan)
        max_raan    = np.deg2rad(self.walker.max_raan)
        inclination = np.deg2rad(self.walker.inclination)
        altitude    = self.walker.altitude * 1000
        sat_count   = self.get_total_sat_count()

        raans = np.linspace(start_raan, start_raan + max_raan, self.walker.plane_count + 1)
        raans = raans[:-1] % (2 * np.pi)

        elements = np.zeros((sat_count, 6))
        idx = 0

        for raan_idx, raan in enumerate(raans):
            for sat_idx in range(self.walker.sats_per_plane):
                sma = CONST.EARTH_RADIUS + altitude
                aol = 2 * np.pi * (sat_idx / self.walker.sats_per_plane + self.walker.f * raan_idx / sat_count)

                elements[idx, :] = [sma, 0, 0, raan, inclination, aol]
                idx += 1

        return elements


class Constellation:
    """
    Расчет группировки
    """
    def __init__(self, config_file: str, name_code: str):
        self.total_sat_count = 0
        self.groups   = []
        self.elements = []
        self.state_eci = []
        self.state_sph = []
        self.load_from_config(config_file, name_code)

    def load_from_config(self, config_file: str, name_code: str):
        """
        Загрузка данных группировки из конфигурационного файла
        :param config_file: имя файла
        :param name_code: имя группировки
        """

        json_data = None
        try:
            with open(config_file) as f:
                json_data = json.load(f)
        except FileNotFoundError:
            raise ConstellationConfigError(f'Файл {config_file} не найден')
        except json.JSONDecodeError as e:
            raise ConstellationConfigError(f'Ошибка чтения JSON: {e}')
        if not json_data:
            raise Constellation404Error(f'Файл {config_file} пустой')

        constellation_data = None
        for group in json_data:
            if group['name'].lower() == name_code.lower():
                constellation_data = group
                print(f'Загружена группировка {name_code}')
                break

        if not constellation_data:
            raise Constellation404Error(f'Группировка {name_code} не найдена')

        try:
            for single_group in constellation_data['Walkers']:
                single_group = WalkerGroup(*single_group)
                self.groups.append(single_group)
                self.total_sat_count += single_group.get_total_sat_count()
        except (KeyError, TypeError, ValueError) as e:
            raise ConstellationConfigError(f'Ошибка в структуре конфигурации: {e}')

    def get_initial_state(self):
        """
        Вычисление элементов орбиты для всех КА в начальный момент
        """

        self.elements = np.zeros((self.total_sat_count, 6))
        shift = 0

        for single_group in self.groups:
            ending = shift + single_group.get_total_sat_count()
            self.elements[shift:ending, :] = single_group.get_initial_elements()
            shift = ending

    def predict(self, epochs: list):
        """
        Расчёт положений всех КА в заданные моменты времени
        :param epochs: моменты времени для расчета, сек
        """

        self.state_eci = np.zeros((self.total_sat_count, 3, len(epochs)))
        self.state_sph = np.zeros((self.total_sat_count, 2, len(epochs)))

        inclination = self.elements[:, 4]
        sma = self.elements[:, 0]
        raan_0 = self.elements[:, 3]
        aol_0 = self.elements[:, 5]

        # Угловая скорость прецессии долготы восходящего узла
        raan_precession_rate = -1.5 * (CONST.EARTH_J2 * np.sqrt(CONST.EARTH_GM) * CONST.EARTH_RADIUS ** 2) \
                               / (sma ** (7/2)) * np.cos(inclination)

        # Угловая скорость аппарата (от восходящего узла, с учетом прецессии)
        draconian_omega = np.sqrt(CONST.EARTH_GM / sma ** 3) \
                         * (1 - 1.5 * CONST.EARTH_J2 * (CONST.EARTH_RADIUS / sma) ** 2) \
                         * (1 - 4 * np.cos(inclination) ** 2)

        for epoch_idx, epoch in enumerate(epochs):
            # Учет смещения
            aol = aol_0 + epoch * draconian_omega
            raan = raan_0 + epoch * raan_precession_rate

            # Переход из сферических в декартову систему координат
            epoch_state = sma * [
                (np.cos(aol) * np.cos(raan) - np.sin(aol) * np.cos(inclination) * np.sin(raan)),
                (np.cos(aol) * np.sin(raan) + np.sin(aol) * np.cos(inclination) * np.cos(raan)),
                (np.sin(aol) * np.sin(inclination))
            ]

            epoch_sph_state = [
                np.arcsin(np.sin(aol) * np.sin(inclination)),
                raan + np.arctan(np.tan(aol) * np.cos(inclination)) + (aol > np.pi) * np.pi
            ]

            self.state_eci[:, :, epoch_idx] = np.array(epoch_state).T
            self.state_sph[:, :, epoch_idx] = np.array(epoch_sph_state).T

    def verify_globality(self, fov, tol=3., epoch=0):
        """
        Алгоритм: для каждой еще не покрытой точки сетки на поверхности Земли (сферы) проверяем покрытие каким-либо спутником

        :param fov: поле зрения каждого аппарата группировки, градусы
        :param tol: шаг сетки
        :param epoch: эпоха на которую проводятся вычисления
        :return: истинность утверждения о глобальности покрытия на эпоху
        """

        fov *= np.pi / 180
        smas = self.elements[:, 0]

        area_size = tol                                                       # угловой размер одной ячейки, градусы
        n_points = round(4 * np.pi * (180 / np.pi) ** 2 / area_size ** 2)     # число точек сетки
        grid_points = self.grid_fibonacci(n_points)                           # генерация точек по сетке Фибоначчи

        theta = np.zeros_like(smas)

        # Расчет зоны обзора каждого спутника
        for idx in range(len(smas)):
            sma = smas[idx]
            arg_sin = np.sin(fov) * (sma / CONST.EARTH_RADIUS)
            if abs(arg_sin) > 1:
                theta[idx] = np.arccos(CONST.EARTH_RADIUS / sma)
            else:
                theta[idx] = np.arcsin(arg_sin) - fov

        # Проверка покрытия каждой точки каждым спутником с отбрасыванием уже покрытых точек
        observed = np.zeros(grid_points.shape[1])
        sat_idx = 0
        while not all(observed) and sat_idx < self.total_sat_count:
            under_sat = self.state_sph[sat_idx, :, epoch]
            for point_idx in np.where(observed == 0)[0]:
                point = grid_points[:, point_idx]
                if self.distance(point, under_sat) < theta[sat_idx]:
                    observed[point_idx] = True

            sat_idx += 1

        return all(observed)

    def visualize_coverage(self, fov, epoch=0):
        """
        Визуализация спутников и границ их зон покрытия на сфере

        :param fov: поле зрения каждого аппарата группировки, градусы
        :param epoch: эпоха на которую проводятся вычисления
        """

        points = self.state_sph[:, :, epoch]
        fov *= np.pi / 180
        smas = self.elements[:, 0]
        coverage = np.zeros_like(smas)

        # Расчет зоны обзора каждого спутника
        for idx in range(len(smas)):
            sma = smas[idx]
            arg_sin = np.sin(fov) * (sma / CONST.EARTH_RADIUS)
            if abs(arg_sin) > 1:
                coverage[idx] = np.arccos(CONST.EARTH_RADIUS / sma)
            else:
                coverage[idx] = np.arcsin(arg_sin) - fov

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        phi, theta = points[:, 0], points[:, 1]
        x_center = np.cos(theta) * np.cos(phi)
        y_center = np.sin(theta) * np.cos(phi)
        z_center = np.sin(phi)


        ax.scatter(x_center, y_center, z_center, color='red', s=5, alpha=0.5, label='Satellites')
        all_polygons = []

        for i in range(len(points)):
            center_vec = np.array([x_center[i], y_center[i], z_center[i]])

            n_points_circle = 20  # количество точек для аппроксимации окружности
            angles = np.linspace(0, 2 * np.pi, n_points_circle)

            boundary_points = []

            for angle in angles:
                # Касательная плоскость с базисом tangent_u tangent_v
                if abs(center_vec[2]) > 0.95:
                    tangent_u = np.array([1, 0, 0])
                else:
                    tangent_u = np.array([-center_vec[1], center_vec[0], 0])
                    tangent_u = tangent_u / np.linalg.norm(tangent_u)

                tangent_v = np.cross(center_vec, tangent_u)
                tangent_v = tangent_v / np.linalg.norm(tangent_v)

                direction = tangent_u * np.cos(angle) + tangent_v * np.sin(angle)

                # Формула поворота Родрига с учетом перпендикулярности поворачиваемого вектора и оси поворота
                cos_fov = np.cos(coverage[i])
                sin_fov = np.sin(coverage[i])

                boundary_point = (center_vec * cos_fov + np.cross(direction, center_vec) * sin_fov)

                boundary_points.append(boundary_point / np.linalg.norm(boundary_point))

            boundary_points = np.array(boundary_points)

            ax.plot(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                    'b-', linewidth=2, alpha=0.1)

            all_polygons.append(boundary_points)

        collection = Poly3DCollection(all_polygons,
                                      alpha=0.05,
                                      facecolors='blue',
                                      linewidths=0,
                                      edgecolors='none')
        ax.add_collection3d(collection)

        ax.grid(False); ax.axis(False); ax.set_aspect('equal')
        ax.set_title(f'Покрытие при FOV = {(180.0 / np.pi * fov):.2f}')

        # Устанавливаем одинаковые пределы для осей
        limit = 1.05
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])

        plt.legend()
        plt.tight_layout()
        plt.show()

    def search_fov_threshold(self, tol=3., epoch=0):
        """
        Бинарный поиск оптимального поля зрения для обеспечения глобальности покрытия с заданной точностью
        :param epoch: эпоха на которую проводятся вычисления
        :param tol: точность определения порога, градусы
        :return: пороговое значение поля зрения
        """
        left = 0
        smas = self.elements[:, 0]
        right = np.max(np.arcsin(CONST.EARTH_RADIUS / smas))
        tol *= np.pi / 180

        if not self.verify_globality(90, 180. / np.pi * tol, epoch):
            return -1

        while (right - left) > tol / 2:
            mid = (right + left) / 2
            if self.verify_globality(180. / np.pi * mid, 180. / np.pi * tol, epoch):
                right = mid
            else:
                left = mid

        return right

    def distance(self, point1, point2):
        """ Угловое расстояние между двумя точками по сфере"""
        phi1, theta1 = point1
        phi2, theta2 = point2
        return np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2))

    def grid_fibonacci(self, n_points):
        """ Сетка Фибоначчи на сфере """
        array_phi, array_theta = [], []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n_points):
            theta = 2 * np.pi * i * golden_ratio            # долгота
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)   # полярный угол

            array_phi.append(np.pi / 2 - phi)               # широта
            array_theta.append(theta % (2 * np.pi))         # долгота

        return np.concatenate([[array_phi], [array_theta]], axis=0)


""" Классы обработки ошибок """
class ConstellationError(Exception):
    pass


class ConstellationConfigError(ConstellationError):
    pass


class Constellation404Error(ConstellationError):
    pass


class ConstellationDataError(ConstellationError):
    pass