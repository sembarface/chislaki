import numpy as np
import matplotlib.pyplot as plt

# Чтение данных из файла
data = np.loadtxt('data.txt')
x_data = data[:, 0]
y_data = data[:, 1]

# Создание объекта для сглаживания
class LocalSmoothing:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
    
    def smooth_point(self, i, polynomial_degree, window_size):
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(self.n, i + half_window + 1)
        
        if start_idx == 0:
            end_idx = min(self.n, window_size + 1)
        elif end_idx == self.n:
            start_idx = max(0, self.n - window_size - 1)
        
        x_window = self.x[start_idx:end_idx]
        y_window = self.y[start_idx:end_idx]
        coefficients = np.polyfit(x_window, y_window, polynomial_degree)
        return np.polyval(coefficients, self.x[i])
    
    def smooth(self, polynomial_degree, window_size):
        smoothed_values = self.y.copy()
        new_values = np.zeros_like(smoothed_values)
        for i in range(self.n):
            new_values[i] = self.smooth_point(i, polynomial_degree, window_size)
        return new_values

# Выполнение сглаживания
smoother = LocalSmoothing(x_data, y_data)
y_smoothed = smoother.smooth(polynomial_degree=2, window_size=6)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', alpha=0.7, label='Исходные данные', s=30)
plt.plot(x_data, y_smoothed, 'blue', linewidth=2, label='Сглаженная функция')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Локальное сглаживание данных')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()