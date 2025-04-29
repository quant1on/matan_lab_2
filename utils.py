import numpy as np
import typing as tp
from random import uniform
import matplotlib.pyplot as plt

def congruent_part(a: float = 0, b: float = 1, n: int = 4) -> tp.Generator[float, None, None]:
    """Построить конгуентное разбиение отрезка

    Args:
        a (float, optional): начало отрезка. Defaults to 0.
        b (float, optional): конец отрезка. Defaults to 1.
        n (int, optional): кол-во разбиений отрезка. Defaults to 4.

    Raises:
        ValueError: в случае невалидного n

    Yields:
        Iterator[tp.Generator[float]]: итератор точек отрезка
    """
    if (n <= 0): raise ValueError("Invalid num of partitions")

    gap = (b - a) / n
    curr = a

    c = 0
    while c <= n:
        yield curr

        c += 1
        curr += gap



def rectangle_method(f: tp.Callable, a: float = 0, b: float = 1,
                     n: int = 4, mode: tp.Literal['L', 'R', 'M', 'rand'] = 'L', partition_iter: tp.Generator = congruent_part) -> float:
    """Посчитать интегральную сумму методом прямоугольников

    Args:
        f (tp.Callable): интегрируемая функция
        a (float, optional): левая граница. Defaults to 0.
        b (float, optional): правая граница. Defaults to 1.
        n (int, optional): кол-во партиций. Defaults to 4.
        mode (tp.Literal[L, R, M, rand], optional): вид взятия значения функции в партиции. Defaults to 'L'.
        partition_iter (tp.Generator, optional): генератор границ подотрезков.

    Raises:
        ValueError: в случае невалидного mode

    Returns:
        float: интегральная сумма
    """
    match(mode):
        case 'L':
            choose_func = lambda x, y: x
        case 'R':
            choose_func = lambda x, y: y
        case 'M':
            choose_func = lambda x, y: (x + y)/2
        case 'rand':
            choose_func = lambda x, y: uniform(x, y)
        case _:
            raise ValueError("Invalid mode")
        
    integ_sum = 0
    part_iter = congruent_part(a, b, n)
    x_r = next(part_iter)

    for i in part_iter:
        x_l = x_r
        x_r = i

        integ_sum += (x_r - x_l) * f(choose_func(x_l, x_r))
    
    return integ_sum

def trapezoid_method(f: tp.Callable, a: float = 0, b: float = 1, n: int = 4, partition_iter: tp.Generator = congruent_part) -> float:
    """Посчитать интегральную сумма методом трапеций

    Args:
        f (tp.Callable): интегрируемая функция.
        a (float, optional): левая граница. Defaults to 0.
        b (float, optional): правая граница. Defaults to 1.
        n (int, optional): кол-во партиций. Defaults to 4.
        partition_iter (tp.Generator, optional): генератор границ подотрезков.

    Returns:
        float: интегральная сумма
    """
    integ_sum = 0
    part_iter = partition_iter(a, b, n)
    x_r = next(part_iter)

    for i in part_iter:
        x_l = x_r
        x_r = i

        integ_sum += 0.5 * (x_r - x_l) * (f(x_l) + f(x_r))
    
    return integ_sum

def simpson_method(f: tp.Callable, a: float = 0, b: float = 1, n: int = 4, partition_iter: tp.Generator = congruent_part) -> float:
    """Посчитать интегральную сумма методом Симпсона

    Args:
        f (tp.Callable): интегрируемая функция.
        a (float, optional): левая граница. Defaults to 0.
        b (float, optional): правая граница. Defaults to 1.
        n (int, optional): кол-во партиций. Defaults to 4.
        partition_iter (tp.Generator, optional): генератор границ подотрезков.

    Returns:
        float: интегральная сумма
    """

    integ_sum = 0
    part_iter = partition_iter(a, b, n)
    x_r = next(part_iter)

    for i in part_iter:
        x_l = x_r
        x_r = i

        integ_sum += (x_r - x_l) * ( f(x_l) + f(x_r) + 4 * f((x_l + x_r)/2) ) / 6
    
    return integ_sum

def plot_comparison(f: tp.Callable, a: float = 0, b: float = 1, n: int = 4,
                    partition_iter: tp.Generator = congruent_part, acc: int = 100):
    
    part_iter = partition_iter(a, b, n)
    xs = np.linspace(a, b, n * acc)

    ys = [f(x) for x in xs]

    fig, ax = plt.subplots(1, 4, figsize=(15, 6))

    ax[0].plot(xs, ys, label='f(x)', color='black')
    ax[0].set_title('Метод прямоугольников (левые)')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    
    ax[1].plot(xs, ys, label='f(x)', color='black')
    ax[1].set_title('Метод прямоугольников (середина)')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')

    ax[2].plot(xs, ys, label='f(x)', color='black')
    ax[2].set_title('Метод трапеций')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')

    ax[3].plot(xs, ys, label='f(x)', color='black')
    ax[3].set_title('Метод Симпсона')
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('y')
    
    x_r = next(part_iter)

    for i in part_iter:
        x_l = x_r
        x_r = i
        x_m = (x_l + x_r) / 2

        ax[0].fill_between([x_l, x_r], [f(x_l), f(x_l)], color='purple', alpha=0.3)
        ax[1].fill_between([x_l, x_r], [f(x_m), f(x_m)], color='orange', alpha=0.3)
        ax[2].fill_between([x_l, x_r], [f(x_l), f(x_r)], color='blue', alpha=0.3)

        x_simp = np.linspace(x_l, x_r, 100)
        y_approx = np.polyval(np.polyfit([x_l, x_m, x_r], [f(x_l), f(x_m), f(x_r)], 2), x_simp)

        ax[3].fill_between(x_simp, y_approx, color='green', alpha=0.3)
    
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    return fig, ax