from math import *
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.sin(x) + 2**x - 5*np.cos(x**2)
    '''return (4/(x**2+1))'''

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

# ВЫЧИСЛЕНИЕ 1: Интерполяция на всем отрезке
integral_approx1 = 0
for k in range(n+1):
    integral_approx1 += C[k] * f(nodes[k])

# ВЫЧИСЛЕНИЕ 2: Составная формула (разбиение на много отрезков)
num_segments = 10  # количество отрезков разбиения
segment_width = (b - a) / num_segments
integral_approx2 = 0

# Узлы для составной формулы
for seg in range(num_segments):
    seg_a = a + seg * segment_width
    seg_b = seg_a + segment_width
    seg_h = segment_width / n
    seg_nodes = [seg_a + i*seg_h for i in range(n+1)]
    
    # Коэффициенты для текущего отрезка
    seg_C = [
        41*seg_h/140,
        216*seg_h/140, 
        27*seg_h/140,
        272*seg_h/140,
        27*seg_h/140,
        216*seg_h/140,
        41*seg_h/140
    ]
    
    # Интеграл на текущем отрезке
    for k in range(n+1):
        integral_approx2 += seg_C[k] * f(seg_nodes[k])

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

# Функция для составной интерполяции (для графика)
def composite_lagrange_poly(x):
    # Находим, к какому отрезку относится x
    seg_idx = int((x - a) / segment_width)
    seg_idx = min(seg_idx, num_segments - 1)
    
    seg_a = a + seg_idx * segment_width
    seg_b = seg_a + segment_width
    seg_h = segment_width / n
    seg_nodes = [seg_a + i*seg_h for i in range(n+1)]
    
    # Вычисляем значение в точке x с помощью интерполяции на этом отрезке
    result = 0
    for k in range(n+1):
        term = f(seg_nodes[k])
        for j in range(n+1):
            if j != k:
                term *= (x - seg_nodes[j]) / (seg_nodes[k] - seg_nodes[j])
        result += term
    return result

print("Квадратурная формула интерполяционного типа")
print(f"Отрезок интегрирования: [{a}, {b}]")
print(f"Степень многочлена Лагранжа: n = {n}")
print(f"Шаг: h = {h:.4f}")
print()
print(f"Приближенное значение интеграла (одна интерполяция): {integral_approx1:.10f}")
print(f"Приближенное значение интеграла (составная формула, {num_segments} отрезков): {integral_approx2:.10f}")
print(f"Разница: {abs(integral_approx1 - integral_approx2):.2e}")

# Построение графиков
x_plot = np.linspace(a, b, 1000)
y_original = [f(x) for x in x_plot]
y_interp1 = [lagrange_poly(x) for x in x_plot]
y_interp2 = [composite_lagrange_poly(x) for x in x_plot]

plt.figure(figsize=(14, 6))

# График 1: Одна интерполяция на всем отрезке
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_original, 'b-', linewidth=2, label='f(x)')
plt.plot(x_plot, y_interp1, 'r--', linewidth=2, label=f'Интерполяция степени {n}')
plt.plot(nodes, [f(x) for x in nodes], 'ko', markersize=6, label='Узлы интерполяции')
plt.fill_between(x_plot, y_original, alpha=0.2, color='blue', label='Площадь под f(x)')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=a, color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=b, color='gray', linestyle=':', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Одна интерполяция на всем отрезке\nИнтеграл = {integral_approx1:.8f}')
plt.legend()

# График 2: Составная интерполяция
plt.subplot(1, 2, 2)
plt.plot(x_plot, y_original, 'b-', linewidth=2, label='f(x)')
plt.plot(x_plot, y_interp2, 'g--', linewidth=2, label=f'Составная интерполяция ({num_segments} отрезков)')

# Отмечаем границы отрезков для составной формулы
for i in range(num_segments + 1):
    x_seg = a + i * segment_width
    plt.axvline(x=x_seg, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    
    # Отмечаем узлы на каждом отрезке
    if i < num_segments:
        seg_a = a + i * segment_width
        seg_h = segment_width / n
        seg_nodes = [seg_a + j*seg_h for j in range(n+1)]
        plt.plot(seg_nodes, [f(x) for x in seg_nodes], 'go', markersize=4, alpha=0.6, label='Узлы интерполяции' if i == 0 else "")

plt.fill_between(x_plot, y_original, alpha=0.2, color='blue', label='Площадь под f(x)')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Составная интерполяция\nИнтеграл = {integral_approx2:.8f}')
plt.legend()

plt.tight_layout()
plt.show()
