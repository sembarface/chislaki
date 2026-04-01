import numpy as np
import matplotlib.pyplot as plt

# Точное решение тестовой задачи

# def p(x):
#     return 1.0 + x

# def f(x):
#     return 2.0 - x*np.exp(x) - x**2 - x**3

# a = 0.0
# b = 1.0

# alpha1 = 2.0
# alpha2 = -1.0

# beta1 = -1.0
# beta2 = 2.0*np.e + 3.0

# def y_exact(x):
#     return np.exp(x) + x**2


def p(x):
    return 1.0 + x**2


def f(x):
    return -2.0*np.exp(x)*np.sin(x) - (1.0 + x**2)*np.exp(x)*np.cos(x)

a = 0.0
b = 1.0

alpha1 = 1.0
beta1 = 0.0

alpha2 = 0.0
beta2 = np.e * (np.cos(1.0) - np.sin(1.0))

def y_exact(x):
    return np.exp(x) * np.cos(x)

# Параметры краевой задачи
# y'(a) - alpha1*y(a) = alpha2
# y'(b) - beta1*y(b) = beta2

# Решение краевой задачи:
# внутри O(h^6), границы O(h^3)
# трёхдиагональная схема

def solve_bvp_num6_bc3(a, b, h, p, f, alpha1, alpha2, beta1, beta2):
    N = int(np.round((b - a) / h))

    if N < 2:
        raise ValueError('Слишком крупный шаг h')

    if abs(a + N*h - b) > 1e-12:
        raise ValueError('Шаг h не делит отрезок [a, b] на целое число частей')

    x = a + h*np.arange(N + 1)

    p_val = p(x)
    f_val = f(x)

    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    dp0 = (p_val[1] - p_val[0]) / h
    df0 = (f_val[1] - f_val[0]) / h

    dpN = (p_val[N] - p_val[N - 1]) / h
    dfN = (f_val[N] - f_val[N - 1]) / h

    # Левая граница: O(h^3)
    #
    # (y1 - y0)/h - alpha1*y0 - alpha2
    # - h/2 * (p0*y0 + f0)
    # - h^2/6 * ((p0*alpha1 + p'(a))*y0 + p0*alpha2 + f'(a)) = 0

    p0 = p_val[0]
    f0 = f_val[0]

    A[0, 0] = -1.0/h - alpha1 - 0.5*h*p0 - (h**2)*(p0*alpha1 + dp0)/6.0
    A[0, 1] =  1.0/h

    rhs[0] = alpha2 + 0.5*h*f0 + (h**2)*(p0*alpha2 + df0)/6.0

    # Внутренние узлы: O(h^6)
    #
    # (1/h^2 - p_{i-1}/12) y_{i-1}
    # + (-2/h^2 - 5p_i/6) y_i
    # + (1/h^2 - p_{i+1}/12) y_{i+1}
    # =
    # (f_{i-1} + 10f_i + f_{i+1}) / 12

    for i in range(1, N):
        A[i, i - 1] =  1.0/h**2 - p_val[i - 1]/12.0
        A[i, i]     = -2.0/h**2 - 5.0*p_val[i]/6.0
        A[i, i + 1] =  1.0/h**2 - p_val[i + 1]/12.0

        rhs[i] = (f_val[i - 1] + 10.0*f_val[i] + f_val[i + 1]) / 12.0

    # Правая граница: O(h^3)
    #
    # (yN - yN-1)/h - beta1*yN - beta2
    # + h/2 * (pN*yN + fN)
    # - h^2/6 * ((pN*beta1 + p'(b))*yN + pN*beta2 + f'(b)) = 0

    pN = p_val[N]
    fN = f_val[N]

    A[N, N - 1] = -1.0/h
    A[N, N] = 1.0/h - beta1 + 0.5*h*pN - (h**2)*(pN*beta1 + dpN)/6.0

    rhs[N] = beta2 - 0.5*h*fN + (h**2)*(pN*beta2 + dfN)/6.0

    y = np.linalg.solve(A, rhs)

    return x, y, A, rhs

# Ошибки на сетке

def errors_on_grid(x, y_num, y_exact):
    y_ex = y_exact(x)
    e = y_num - y_ex

    err_max = np.max(np.abs(e))
    err_l2 = np.linalg.norm(e)

    return y_ex, e, err_max, err_l2


# Решения на двух шагах

h = 0.1

x_h, y_h, A_h, rhs_h = solve_bvp_num6_bc3(
    a, b, h,
    p, f,
    alpha1, alpha2, beta1, beta2
)

x_h2, y_h2, A_h2, rhs_h2 = solve_bvp_num6_bc3(
    a, b, h/2.0,
    p, f,
    alpha1, alpha2, beta1, beta2
)


# Ошибки

y_ex_h, e_h, emax_h, el2_h = errors_on_grid(x_h, y_h, y_exact)
y_ex_h2, e_h2, emax_h2, el2_h2 = errors_on_grid(x_h2, y_h2, y_exact)


# Фактический порядок

p_max = np.log2(emax_h / emax_h2)
p_l2 = np.log2(el2_h / el2_h2)

print("=== Ошибки ===")
print(f"h   = {h}   max|e| = {emax_h:.6e}   L2 = {el2_h:.6e}")
print(f"h/2 = {h/2} max|e| = {emax_h2:.6e}   L2 = {el2_h2:.6e}")

print("\n=== Фактический порядок ===")
print("p_max =", p_max)
print("p_L2  =", p_l2)


# График решения

plt.figure(figsize=(8, 5))
plt.plot(x_h2, y_ex_h2, label="exact y")
plt.plot(x_h,  y_h, "o-", label="num y (h)")
plt.plot(x_h2, y_h2, ".-", label="num y (h/2)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Точное и приближенное решение")
plt.grid(True)
plt.legend()


# График ошибки

plt.figure(figsize=(8, 5))
plt.plot(x_h,  np.abs(e_h), "o-", label="|error| (h)")
plt.plot(x_h2, np.abs(e_h2), ".-", label="|error| (h/2)")

plt.yscale("log")
plt.xlabel("x")
plt.ylabel("error")
plt.title("Поточечные ошибки")
plt.grid(True, which="both")
plt.legend()


# Сходимость на последовательности шагов

h_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
err_list = []

for h_cur in h_list:
    x_cur, y_cur, A_cur, rhs_cur = solve_bvp_num6_bc3(
        a, b, h_cur,
        p, f,
        alpha1, alpha2, beta1, beta2
    )

    y_ex_cur, e_cur, emax_cur, el2_cur = errors_on_grid(x_cur, y_cur, y_exact)
    err_list.append(emax_cur)

err_list = np.array(err_list)

# print("\n=== Таблица погрешностей ===")
# print("h           max|e|         ratio")
# for i in range(len(h_list)):
#     if i == 0:
#         print(f"{h_list[i]:<10.5f} {err_list[i]:<14.6e} ---")
#     else:
#         ratio = err_list[i - 1] / err_list[i]
#         print(f"{h_list[i]:<10.5f} {err_list[i]:<14.6e} {ratio:.6f}")

print("\n=== Наблюдаемый порядок ===")
for i in range(1, len(h_list)):
    p_obs = np.log2(err_list[i - 1] / err_list[i])
    print(f"h = {h_list[i - 1]:.5f} -> {h_list[i]:.5f}   p = {p_obs:.6f}")


# График сходимости

plt.figure(figsize=(8, 5))
plt.plot(h_list, err_list, "o-", label="max|e|")

c = err_list[0] / (h_list[0]**3)
ref = c * np.array(h_list)**3
plt.plot(h_list, ref, ".-", label="C*h^3")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("h")
plt.ylabel("max error")
plt.title("Сходимость")
plt.grid(True, which="both")
plt.legend()

plt.show()