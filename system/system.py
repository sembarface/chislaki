from math import *
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(9, 6))
eps = 1e-10


def f(x, y):
    '''Второе уравнение системы: g(x,y) = 0'''
    '''return x**2 - y - 1'''
    return 7*x**2 - 4*y

def g(x, y):
    '''Первое уравнение системы: f(x,y) = 0'''
    '''return x**2 + y**2 - 4'''
    return np.sin(x)+ (y-3)**3

def f_x(x,y):
    return (f(x+eps, y) - f(x-eps, y)) / (2*eps)

def f_y(x,y):
    return (f(x, y+eps) - f(x, y-eps)) / (2*eps)

def g_x(x,y):
    return (g(x+eps, y) - g(x-eps, y)) / (2*eps)

def g_y(x,y):
    return (g(x, y+eps) - g(x, y-eps)) / (2*eps)

# Метод Брауна
x0 = 10
y0 = 10
x_k = x0
y_k = y0
c = 0
p_k=q_k=1
while max(abs(q_k), abs(p_k)) >= eps:
    # Шаг 1: Вычисление x_volna
    x_volna = x_k -(f(x_k, y_k))/(f_x(x_k, y_k))
    
    # Шаг 2: Вычисление q_k
    q_k = (g(x_volna, y_k)*f_x(x_k,y_k)) / (f_x(x_k, y_k)*g_y(x_k, y_k) - f_y(x_k, y_k)*g_x(x_volna, y_k))
    
    # Шаг 3: Вычисление p_k
    p_k = (f(x_k, y_k) - q_k * f_y(x_k, y_k)) / (f_x(x_k, y_k))
    
    # Шаг 4: Новое приближение
    x = x_k - p_k
    y = y_k - q_k
    
    
    x_k = x
    y_k = y
    c += 1
    plt.plot(x, y, 'yo', markersize=7)

plt.plot(x, y, 'yo',label=f'Сходящиеся точки')
print('Метод Брауна для системы уравнений')
print(f'Ответ: x = {x}, y = {y}')
print(f'Итераций: {c}')
print(f'Проверка: f(x,y) = {f(x, y)}, g(x,y) = {g(x, y)}')

# Построение графиков
x_vals = np.linspace(-25, 25, 4000)
y_vals = np.linspace(-25, 25, 4000)
X, Y = np.meshgrid(x_vals, y_vals)

# Вычисление значений функций
F = f(X, Y)
G = g(X, Y)

# Создание графика

# График f(x,y) = 0 (окружность)
plt.contour(X, Y, F, levels=[0], colors='blue', linewidths=2)

# График g(x,y) = 0 (парабола)
plt.contour(X, Y, G, levels=[0], colors='red', linewidths=2)

# Точка пересечения (решение)
plt.plot(x, y, 'ko', markersize=8, label=f'Решение: ({x:.4f}, {y:.4f})')

# Оси координат
plt.axhline(y=0, color='black', linewidth=0.5, alpha=0.7)
plt.axvline(x=0, color='black', linewidth=0.5, alpha=0.7)

# Настройки графика
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графическое решение системы уравнений методом Брауна')
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Добавляем начальное приближение
plt.plot(x0, y0, 'go', markersize=6, label=f'Начальное приближение: ({x0}, {y0})')

plt.legend()
plt.show()
'''https://ikfia.ysn.ru/wp-content/uploads/2018/01/OrtegaRejnboldt1975ru.pdf'''
