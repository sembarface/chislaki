import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)
x = np.arange(0, 9.0, 0.01)  # 900 точек с шагом 0.01

# Базовая синусоида с несимметричными амплитудами
base_signal = 2 * np.sin(2 * x) + 1.5 * np.cos(0.5 * x)

# Добавляем сильный несимметричный шум
noise_positive = np.random.exponential(1.5, len(x))  # Экспоненциальный шум для положительной части
noise_negative = -np.random.exponential(2.0, len(x))  # Более сильный шум для отрицательной части

# Комбинируем шум в зависимости от значения сигнала
combined_noise = np.where(base_signal > 0, noise_positive, noise_negative)

# Добавляем случайные выбросы
outliers = np.random.choice([-1, 0, 1], size=len(x), p=[0.05, 0.9, 0.05]) * np.random.uniform(3, 8, len(x))

# Финальный сигнал с шумом и выбросами
y = base_signal + combined_noise + outliers

# Сохраняем в файл
with open('data.txt', 'w') as f:
    for i in range(len(x)):
        f.write(f"{x[i]:.2f} {y[i]:.6f}\n")

print("Файл data.txt создан с 900 точками")