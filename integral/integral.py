from math import *
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.sin(x) +2**x -5*np.cos(x**2) # пример функции для интегрирования

# Параметры интегрирования
a = 0
b = 3
n = 6  # степень многочлена Лагранжа

# Равномерно распределенные узлы на отрезке [a, b]
h = (b - a) / n
nodes = [a + i*h for i in range(n+1)]

# Коэффициенты для n=6 (формула 7-го порядка точности)
C = [
    41*h/140,
    216*h/140, 
    27*h/140,
    272*h/140,
    27*h/140,
    216*h/140,
    41*h/140
]

# Вычисление интеграла по квадратурной формуле
integral_approx = 0
for k in range(n+1):
    integral_approx += C[k] * f(nodes[k])

# Функция для вычисления интерполяционного многочлена Лагранжа
def lagrange_poly(x):
    result = 0
    for k in range(n+1):
        term = f(nodes[k])
        for j in range(n+1):
            if j != k:
                term *= (x - nodes[j]) / (nodes[k] - nodes[j])
        result += term
    return result

print("Квадратурная формула интерполяционного типа")
print(f"Отрезок интегрирования: [{a}, {b}]")
print(f"Степень многочлена Лагранжа: n = {n}")
print(f"Шаг: h = {h:.4f}")
print(f"Узлы интерполирования: {[f'{x:.4f}' for x in nodes]}")
print(f"Коэффициенты C_k: {[f'{c:.6f}' for c in C]}")
print()
print(f"Приближенное значение интеграла: {integral_approx:.10f}")

# Построение графиков
x_plot = np.linspace(a, b, 1000)
y_original = [f(x) for x in x_plot]
y_interp = [lagrange_poly(x) for x in x_plot]

plt.figure(figsize=(12, 8))

# График исходной функции
plt.plot(x_plot, y_original, 'b-', linewidth=2, label='f(x)')

# График интерполяционного многочлена
plt.plot(x_plot, y_interp, 'r--', linewidth=2, label=f'Интерполяционный многочлен степени {n}')

# Узлы интерполяции
y_nodes = [f(x) for x in nodes]
plt.plot(nodes, y_nodes, 'ko', markersize=8, label='Узлы интерполяции')

# Закрашивание площади под кривой (интеграл)
x_fill = np.linspace(a, b, 200)
y_fill_original = [f(x) for x in x_fill]
plt.fill_between(x_fill, y_fill_original, alpha=0.2, color='blue', label='Площадь под f(x)')

# Настройки графика
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=a, color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=b, color='gray', linestyle=':', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis('equal')
plt.title(f'Интерполяция функции многочленом Лагранжа степени {n}')
plt.legend()

plt.xlim(a - 1, b + 1)
plt.ylim(a - 1, b + 1)

plt.show()
