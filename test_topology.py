"""
test_topology.py
================
Эталонная реализация fold-отображений для плоских факторпространств
(плоский 3-тор T^3, скрученный тор, зеркальный орбифолд) + тесты-проверки.

Эти же формулы один-в-один переносятся в GLSL-шейдер (topology_shader.html),
поэтому файл одновременно служит:
  * справочной (reference) реализацией математики из ноутбука topology_theory.ipynb;
  * набором автоматических проверок (assert), которые ловят ошибки в формулах.

Запуск:   python test_topology.py     (печатает PASS/FAIL по каждому тесту)
          pytest test_topology.py     (тоже работает — функции test_* совместимы)

Зависимости: только numpy.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
#  Fold-отображения  fold: \tilde X (= R^3) -> фундаментальная область D
#  Топология кодируется группой Gamma: X = \tilde X / Gamma.
# ---------------------------------------------------------------------------

def fold_torus(p, L: float = 4.0):
    """Плоский 3-тор  T^3 = R^3 / (L Z)^3.

    Сворачивает точку(и) в куб-ячейку [-L/2, L/2)^3 чистой трансляцией:
        q = mod(p + L/2, L) - L/2.
    p: array (..., 3).  Возвращает array той же формы.
    """
    p = np.asarray(p, dtype=float)
    return np.mod(p + 0.5 * L, L) - 0.5 * L


def fold_lattice(p, A):
    """Общий тор T^n = R^n / Lambda для произвольной решётки Lambda = A Z^n.

    Базис решётки — СТОЛБЦЫ матрицы A = [a_1 ... a_n].  Точка с координатами
    c в этом базисе есть p = A c, откуда c = A^{-1} p, а ближайший узел решётки
    есть A * round(c).  Поэтому
        q = p - A * round(A^{-1} p).
    Для кубической A = L*I это совпадает с fold_torus.
    p: array (..., n).
    """
    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    Ainv = np.linalg.inv(A)
    coeff = p @ Ainv.T          # (..., n) = (A^{-1} p) в строковой записи
    k = np.round(coeff)
    return p - k @ A.T


def fold_twisted(p, L: float = 4.0):
    """Скрученный 3-тор (mapping torus) с монодромией phi = R_x(pi).

    За каждый шаг на соседнюю ячейку вдоль x содержимое поворачивается на 180
    градусов вокруг оси x:  (y, z) -> (-y, -z).  Индекс ячейки ix = floor((x+L/2)/L);
    при нечётном ix применяем поворот.  phi^2 = I  =>  истинный период вдоль x
    равен 2L.  det phi = +1  =>  пространство ориентируемо.
    """
    p = np.asarray(p, dtype=float)
    q = fold_torus(p, L).copy()
    ix = np.floor((p[..., 0] + 0.5 * L) / L)
    flip = np.mod(ix, 2.0) > 0.5
    sign = np.where(flip, -1.0, 1.0)
    q[..., 1] *= sign
    q[..., 2] *= sign
    return q


def fold_mirror(p, L: float = 4.0):
    """Зеркальный орбифолд: фундаментальная область с отражающими стенками.

    Треугольная волна  q = |mod(p, 2L) - L| - L/2  даёт зеркала на гранях
    ячейки (группа, порождённая отражениями — пример группы Кокстера).
    Это уже НЕ свободное действие: на зеркалах есть неподвижные точки,
    поэтому фактор — орбифолд, а не многообразие.
    """
    p = np.asarray(p, dtype=float)
    return np.abs(np.mod(p, 2.0 * L) - L) - 0.5 * L


# ---------------------------------------------------------------------------
#  Метрика тора: геодезическое расстояние и образы (images)
# ---------------------------------------------------------------------------

def torus_distance(a, b, L: float = 4.0):
    """Геодезическое расстояние на плоском 3-торе.

    d_T(a,b) = min_{n in Z^3} |a - b - L n| = |fold(a - b)|,
    т.к. кратчайший образ разности и есть свёрнутая разность.
    """
    d = fold_torus(np.asarray(a, float) - np.asarray(b, float), L)
    return np.linalg.norm(d, axis=-1)


def star_images(s, L: float = 4.0, n: int = 2):
    """Образы одиночной звезды s в накрытии R^3: { s + L k : k in [-n, n]^3 }.

    Именно это видит наблюдатель в 3-торе — одну звезду как 3D-решётку копий
    ("зеркальный зал" / космическая кристаллография).  Возвращает array (M, 3).
    """
    s = np.asarray(s, dtype=float)
    rng = np.arange(-n, n + 1)
    kx, ky, kz = np.meshgrid(rng, rng, rng, indexing="ij")
    k = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=-1)  # (M, 3)
    return s[None, :] + L * k


def count_images_within(observer, s, R: float, L: float = 4.0):
    """Сколько образов звезды s попадает в "горизонт" радиуса R вокруг наблюдателя.

    Для большого R число образов ~ объём шара / объём ячейки = (4/3 pi R^3)/L^3
    (плотность ровно 1 образ на ячейку).  Используется в тесте плотности.
    """
    nmax = int(np.ceil(R / L)) + 1
    imgs = star_images(s, L=L, n=nmax)
    dist = np.linalg.norm(imgs - np.asarray(observer, float)[None, :], axis=-1)
    return int(np.sum(dist <= R))


# ===========================================================================
#  ТЕСТЫ
# ===========================================================================
RNG = np.random.default_rng(0)
EPS = 1e-9


def test_fold_in_domain():
    """fold_torus загоняет любую точку в куб [-L/2, L/2]."""
    L = 4.0
    p = RNG.uniform(-100, 100, size=(2000, 3))
    q = fold_torus(p, L)
    assert np.all(q <= 0.5 * L + 1e-9) and np.all(q >= -0.5 * L - 1e-9)


def test_fold_periodic():
    """fold(p + L*n) == fold(p) для любого целочисленного сдвига решётки."""
    L = 4.0
    p = RNG.uniform(-10, 10, size=(2000, 3))
    n = RNG.integers(-5, 6, size=(2000, 3))
    assert np.allclose(fold_torus(p + L * n, L), fold_torus(p, L), atol=1e-9)


def test_fold_idempotent():
    """fold(fold(p)) == fold(p) — точка уже в области не двигается."""
    L = 4.0
    p = RNG.uniform(-50, 50, size=(2000, 3))
    q = fold_torus(p, L)
    assert np.allclose(fold_torus(q, L), q, atol=1e-9)


def test_general_lattice_matches_cubic():
    """Общий fold_lattice с A = L*I совпадает с кубическим fold_torus."""
    L = 4.0
    A = L * np.eye(3)
    p = RNG.uniform(-30, 30, size=(2000, 3))
    assert np.allclose(fold_lattice(p, A), fold_torus(p, L), atol=1e-9)


def test_general_lattice_periodic():
    """fold_lattice периодичен по решётке Lambda = A Z^3 для косой A."""
    A = np.array([[3.0, 1.0, 0.0],
                  [0.0, 4.0, 1.0],
                  [0.0, 0.0, 5.0]])
    p = RNG.uniform(-20, 20, size=(1500, 3))
    n = RNG.integers(-4, 5, size=(1500, 3))
    shift = n @ A.T                      # вектор решётки A n
    assert np.allclose(fold_lattice(p + shift, A), fold_lattice(p, A), atol=1e-8)


def test_torus_distance_properties():
    """d_T симметрично, неотрицательно и не превосходит евклидова расстояния."""
    L = 4.0
    a = RNG.uniform(-10, 10, size=(2000, 3))
    b = RNG.uniform(-10, 10, size=(2000, 3))
    dT = torus_distance(a, b, L)
    assert np.allclose(dT, torus_distance(b, a, L), atol=1e-9)        # симметрия
    assert np.all(dT >= -1e-12)                                       # неотрицательность
    assert np.all(dT <= np.linalg.norm(a - b, axis=-1) + 1e-9)       # <= евклидова
    assert np.all(dT <= 0.5 * L * np.sqrt(3) + 1e-9)                 # <= радиус ячейки


def test_twisted_period_2L():
    """Скрученный тор: истинный период вдоль x равен 2L (phi^2 = I)."""
    L = 4.0
    p = RNG.uniform(-10, 10, size=(2000, 3))
    shift = np.zeros_like(p); shift[:, 0] = 2.0 * L
    assert np.allclose(fold_twisted(p + shift, L), fold_twisted(p, L), atol=1e-9)


def test_twisted_monodromy():
    """Один шаг по x применяет поворот R_x(pi): (y,z) -> (-y,-z)."""
    L = 4.0
    # берём точки в первой ячейке, чтобы flip не сработал (ix=0 чётный)
    p = RNG.uniform(-0.4 * L, 0.4 * L, size=(1000, 3))
    q0 = fold_twisted(p, L)
    shift = np.zeros_like(p); shift[:, 0] = L          # переход в соседнюю ячейку (ix=1)
    q1 = fold_twisted(p + shift, L)
    expected = q0.copy(); expected[:, 1] *= -1; expected[:, 2] *= -1
    assert np.allclose(q1, expected, atol=1e-9)


def test_mirror_in_domain():
    """Зеркальная свёртка тоже даёт точку в [-L/2, L/2]."""
    L = 4.0
    p = RNG.uniform(-100, 100, size=(2000, 3))
    q = fold_mirror(p, L)
    assert np.all(q <= 0.5 * L + 1e-9) and np.all(q >= -0.5 * L - 1e-9)


def test_mirror_reflection():
    """На границе ячейки зеркало: fold_mirror непрерывна и чётна относительно стенок.

    Проверяем, что отражение точки относительно стенки x=0 даёт тот же образ:
    fold_mirror(p) и fold_mirror(p со знаком -x) совпадают по x-компоненте.
    """
    L = 4.0
    p = RNG.uniform(-50, 50, size=(2000, 3))
    pm = p.copy(); pm[:, 0] *= -1
    assert np.allclose(fold_mirror(p, L)[:, 0], fold_mirror(pm, L)[:, 0], atol=1e-9)


def test_star_images_fold_back():
    """ВСЕ образы одной звезды сворачиваются в одну точку: fold(s + L k) == fold(s).

    Это математическая суть теста "одна звезда в торе": её копии — тот же объект.
    """
    L = 4.0
    s = np.array([1.3, -0.7, 0.4])
    imgs = star_images(s, L=L, n=3)            # (343, 3)
    folded = fold_torus(imgs, L)
    assert np.allclose(folded, fold_torus(s, L)[None, :], atol=1e-9)


def test_image_count_density():
    """Число образов звезды в горизонте R ~ (4/3 pi R^3)/L^3 (плотность 1/ячейку)."""
    L = 4.0
    observer = np.array([0.5, 0.2, -0.3])
    s = np.array([1.0, -1.0, 0.5])
    R = 6.0 * L
    n_actual = count_images_within(observer, s, R, L)
    n_expected = (4.0 / 3.0) * np.pi * R**3 / L**3
    rel_err = abs(n_actual - n_expected) / n_expected
    assert rel_err < 0.15, f"images={n_actual}, expected~{n_expected:.0f}, rel_err={rel_err:.3f}"


def test_one_planet_one_star_scene():
    """Сценарий "одна планета и одна звезда" в 3-торе.

    Планета (наблюдатель) видит звезду по НЕСКОЛЬКИМ геодезическим (обходя тор
    в разные стороны), т.е. как множество образов.  Проверяем две вещи:
      (a) ближайший образ даёт ровно тороидальное расстояние torus_distance;
      (b) при АНТИПОДАЛЬНОМ разносе (сдвиг ровно L/2 по одной оси) есть РОВНО
          два кратчайших пути одинаковой длины — обход тора в "+" и в "-".
    """
    L = 4.0
    # (a) общий случай: минимум по образам == тороидальная метрика
    planet = np.array([-1.8, 0.3, 0.5])
    star   = np.array([ 1.4, -0.6, 0.2])
    dists = np.linalg.norm(star_images(star, L=L, n=3) - planet[None, :], axis=-1)
    assert np.isclose(dists.min(), torus_distance(planet, star, L), atol=1e-9)

    # (b) антиподальная конфигурация: ровно два кратчайших образа длиной L/2
    planet2 = np.array([0.0, 0.0, 0.0])
    star2   = np.array([0.5 * L, 0.0, 0.0])
    d2 = np.linalg.norm(star_images(star2, L=L, n=2) - planet2[None, :], axis=-1)
    dmin = d2.min()
    assert np.isclose(dmin, 0.5 * L)
    assert np.sum(d2 <= dmin + 1e-6) == 2, "ожидали ровно 2 кратчайших образа"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{n_pass}/{len(tests)} тестов прошло.")
    return n_pass == len(tests)


if __name__ == "__main__":
    ok = _run_all()
    raise SystemExit(0 if ok else 1)
