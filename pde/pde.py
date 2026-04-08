import numpy as np
import matplotlib.pyplot as plt


# ТЕСТОВАЯ ЗАДАЧА
# U_t = a^2 U_xx,  0 < x < l,  0 < t <= T
#
# Точное решение:
# U(t, x) = exp(-a^2 * t) * sin(x)
#
# Тогда:
# U_t  = -a^2 * exp(-a^2*t) * sin(x)
# U_xx = -exp(-a^2*t) * sin(x)
# a^2 * U_xx = U_t

# a = 1.0
# l = 6.0
# T = 3.0

# alpha1 = 1.0
# alpha2 = 1.0

# beta1 = 1.0
# beta2 = 1.0


# def u_exact(t, x):
#     return np.exp(-a**2 * t) * np.sin(x)


# def psi(x):
#     return np.sin(x)


# def alpha3(t):
#     return np.exp(-a**2 * t)


# def beta3(t):
#     return np.exp(-a**2 * t) * (np.sin(l) + np.cos(l))






# ТЕСТОВАЯ ЗАДАЧА
# U_t = a^2 U_xx,  0 < x < l,  0 < t <= T
#
# Точное решение:
# U(t, x) = exp(-a^2 * t) * sin(x)
#         + 0.5 * exp(-4 * a^2 * t) * sin(2x)
#
# Тогда:
# U_t  = a^2 U_xx


a = 1.0
l = 6.0
T = 1.0

alpha1 = 1.0
alpha2 = 1.0

beta1 = 1.0
beta2 = 1.0


def u_exact(t, x):
    return (
        np.exp(-a**2 * t) * np.sin(x)
        +
        0.5 * np.exp(-4 * a**2 * t) * np.sin(2*x)
    )


def psi(x):
    return np.sin(x) + 0.5 * np.sin(2*x)


def alpha3(t):
    # x = 0
    # sin(0)=0, sin(2*0)=0
    # cos(0)=1, cos(2*0)=1

    return (
        np.exp(-a**2 * t)
        +
        np.exp(-4 * a**2 * t)
    )


def beta3(t):
    # x = l

    return (
        np.exp(-a**2 * t) * (np.sin(l) + np.cos(l))
        +
        0.5 * np.exp(-4 * a**2 * t) * (np.sin(2*l) + 2*np.cos(2*l))
    )





# ТЕСТОВАЯ ЗАДАЧА
# U_t = a^2 U_xx,  0 < x < l,  0 < t <= T
#
# Точное решение:
# U(t, x) = exp(-a^2 * t) * cos(x)
#
# Тогда:
# U_t  = -a^2 * exp(-a^2*t) * cos(x)
# U_xx = -exp(-a^2*t) * cos(x)
# a^2 * U_xx = U_t


# a = 1.0
# l = 6.0
# T = 3.0

# alpha1 = 1.0
# alpha2 = 1.0

# beta1 = 1.0
# beta2 = 1.0


# def u_exact(t, x):
#     return np.exp(-a**2 * t) * np.cos(x)


# def psi(x):
#     return np.cos(x)


# def alpha3(t):
#     # x = 0:
#     # U(t,0) = exp(-a^2 t)
#     # U_x(t,0) = 0
#     return np.exp(-a**2 * t)


# def beta3(t):
#     # x = l:
#     # U(t,l) = exp(-a^2 t) * cos(l)
#     # U_x(t,l) = -exp(-a^2 t) * sin(l)
#     return np.exp(-a**2 * t) * (np.cos(l) - np.sin(l))







# ПРОГОНКА ДЛЯ ТРЁХДИАГОНАЛЬНОЙ СИСТЕМЫ

def solve_tridiagonal(a_diag, b_diag, c_diag, rhs):
    n = len(b_diag)

    aa = a_diag.copy()
    bb = b_diag.copy()
    cc = c_diag.copy()
    dd = rhs.copy()

    for i in range(1, n):
        m = aa[i] / bb[i - 1]
        bb[i] = bb[i] - m * cc[i - 1]
        dd[i] = dd[i] - m * dd[i - 1]

    y = np.zeros(n)
    y[-1] = dd[-1] / bb[-1]

    for i in range(n - 2, -1, -1):
        y[i] = (dd[i] - cc[i] * y[i + 1]) / bb[i]

    return y


# ПОСТРОЕНИЕ МАТРИЦЫ СХЕМЫ

def build_matrix_heat_robin(l, h, tau, a, alpha1, alpha2, beta1, beta2):
    N = int(np.round(l / h))

    if N < 2:
        raise ValueError('Слишком крупный шаг h')

    if abs(N * h - l) > 1e-12:
        raise ValueError('Шаг h не делит отрезок [0, l] на целое число частей')

    a2 = a**2
    sigma = a2 * tau / h**2

    lower = np.zeros(N + 1)
    diag = np.zeros(N + 1)
    upper = np.zeros(N + 1)

    # Левая граница:
    # (alpha1 - alpha2/h - alpha2*h/(2*a^2*tau)) * U_0^{k+1}
    # + alpha2/h * U_1^{k+1}
    # = alpha3(t_{k+1}) - alpha2*h/(2*a^2*tau) * U_0^k

    diag[0] = alpha1 - alpha2 / h - alpha2 * h / (2.0 * a2 * tau)
    upper[0] = alpha2 / h

    # Внутренние узлы:
    # -sigma * U_{j-1}^{k+1} + (1 + 2*sigma) * U_j^{k+1}
    # -sigma * U_{j+1}^{k+1} = U_j^k

    for j in range(1, N):
        lower[j] = -sigma
        diag[j] = 1.0 + 2.0 * sigma
        upper[j] = -sigma

    # Правая граница:
    # -beta2/h * U_{N-1}^{k+1}
    # + (beta1 + beta2/h + beta2*h/(2*a^2*tau)) * U_N^{k+1}
    # = beta3(t_{k+1}) + beta2*h/(2*a^2*tau) * U_N^k

    lower[N] = -beta2 / h
    diag[N] = beta1 + beta2 / h + beta2 * h / (2.0 * a2 * tau)

    return lower, diag, upper


# ПРАВАЯ ЧАСТЬ НА ОДНОМ ШАГЕ ПО ВРЕМЕНИ

def build_rhs_one_step(u_prev, t_next, h, tau, a, alpha2, beta2, alpha3, beta3):
    N = len(u_prev) - 1
    a2 = a**2

    rhs = np.zeros(N + 1)

    rhs[0] = alpha3(t_next) - alpha2 * h / (2.0 * a2 * tau) * u_prev[0]

    for j in range(1, N):
        rhs[j] = u_prev[j]

    rhs[N] = beta3(t_next) + beta2 * h / (2.0 * a2 * tau) * u_prev[N]

    return rhs


# ЧИСЛЕННОЕ РЕШЕНИЕ

def solve_heat_robin_implicit(l, T, h, tau, a,
                              psi, alpha1, alpha2, alpha3,
                              beta1, beta2, beta3):
    N = int(np.round(l / h))
    K = int(np.round(T / tau))

    if abs(N * h - l) > 1e-12:
        raise ValueError('Шаг h не делит отрезок [0, l] на целое число частей')

    if abs(K * tau - T) > 1e-12:
        raise ValueError('Шаг tau не делит отрезок [0, T] на целое число частей')

    x = h * np.arange(N + 1)
    t = tau * np.arange(K + 1)

    u = np.zeros((K + 1, N + 1))

    # Начальный слой
    u[0, :] = psi(x)

    # Матрица одна и та же на каждом слое
    lower, diag, upper = build_matrix_heat_robin(
        l, h, tau, a,
        alpha1, alpha2, beta1, beta2
    )

    for k in range(K):
        rhs = build_rhs_one_step(
            u[k, :], t[k + 1],
            h, tau, a,
            alpha2, beta2,
            alpha3, beta3
        )

        u[k + 1, :] = solve_tridiagonal(lower, diag, upper, rhs)

    return x, t, u, lower, diag, upper


# ОШИБКИ

def errors_on_grid(x, t, u_num, u_exact):
    K = len(t)
    N = len(x)

    u_ex = np.zeros((K, N))

    for k in range(K):
        u_ex[k, :] = u_exact(t[k], x)

    e = u_num - u_ex

    err_max = np.max(np.abs(e))
    # err_l2 = np.linalg.norm(e)

    return u_ex, e, err_max


# ТЕСТ НА ДВУХ СЕТКАХ
# Поскольку схема имеет порядок O(tau + h^2),
# для наблюдения второго порядка удобно брать tau ~ h^2

h = 0.1
tau = 0.5 * h**2

x_h, t_h, u_h, lower_h, diag_h, upper_h = solve_heat_robin_implicit(
    l, T, h, tau, a,
    psi, alpha1, alpha2, alpha3,
    beta1, beta2, beta3
)

x_h2, t_h2, u_h2, lower_h2, diag_h2, upper_h2 = solve_heat_robin_implicit(
    l, T, h / 2.0, tau / 4.0, a,
    psi, alpha1, alpha2, alpha3,
    beta1, beta2, beta3
)


u_ex_h, e_h, emax_h = errors_on_grid(x_h, t_h, u_h, u_exact)
u_ex_h2, e_h2, emax_h2 = errors_on_grid(x_h2, t_h2, u_h2, u_exact)

p_max = np.log2(emax_h / emax_h2)
# p_l2 = np.log2(el2_h / el2_h2)

print("=== Ошибки на всей сетке (x, t) ===")
print(f"h   = {h:.5f},   tau   = {tau:.5f},   max|e| = {emax_h:.6e}")
print(f"h/2 = {h/2:.5f}, tau/4 = {tau/4:.5f}, max|e| = {emax_h2:.6e}")

print("\n=== Наблюдаемый порядок ===")
print("p_max =", p_max)
# print("p_L2  =", p_l2)


# СХОДИМОСТЬ НА ПОСЛЕДОВАТЕЛЬНОСТИ СЕТОК

# h_list = [0.2, 0.1, 0.05, 0.025]
# err_list = []

# for h_cur in h_list:
#     tau_cur = 0.5 * h_cur**2

#     x_cur, t_cur, u_cur, lower_cur, diag_cur, upper_cur = solve_heat_robin_implicit(
#         l, T, h_cur, tau_cur, a,
#         psi, alpha1, alpha2, alpha3,
#         beta1, beta2, beta3
#     )

#     u_ex_cur, e_cur, emax_cur, el2_cur = errors_on_grid(x_cur, t_cur, u_cur, u_exact)
#     err_list.append(emax_cur)

# err_list = np.array(err_list)

# print("\n=== Таблица сходимости ===")
# print("h           max|e|         p")
# for i in range(len(h_list)):
#     if i == 0:
#         print(f"{h_list[i]:<10.5f} {err_list[i]:<14.6e} ---")
#     else:
#         p_obs = np.log2(err_list[i - 1] / err_list[i])
#         print(f"{h_list[i]:<10.5f} {err_list[i]:<14.6e} {p_obs:.6f}")


# ГРАФИКИ

# Последний временной слой
plt.figure(figsize=(8, 5))
plt.plot(x_h2, u_ex_h2[-1, :], label='exact U(T,x)')
plt.plot(x_h,  u_h[-1, :], 'o-', label='num U(T,x), h')
plt.plot(x_h2, u_h2[-1, :], '.-', label='num U(T,x), h/2')

plt.xlabel('x')
plt.ylabel('U')
plt.title('Решение на последнем временном слое')
plt.grid(True)
plt.legend()

# Ошибка на последнем временном слое
plt.figure(figsize=(8, 5))
plt.plot(x_h,  np.abs(e_h[-1, :]), 'o-', label='|error|(T,x), h')
plt.plot(x_h2, np.abs(e_h2[-1, :]), '.-', label='|error|(T,x), h/2')

plt.yscale('log')
plt.xlabel('x')
plt.ylabel('error')
plt.title('Поточечная ошибка на последнем временном слое')
plt.grid(True, which='both')
plt.legend()

# Сходимость
# plt.figure(figsize=(8, 5))
# plt.plot(h_list, err_list, 'o-', label='max|e|')

# c = err_list[0] / (h_list[0]**2)
# ref = c * np.array(h_list)**2
# plt.plot(h_list, ref, '.-', label='C*h^2')

# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('h')
# plt.ylabel('max error')
# plt.title('Сходимость')
# plt.grid(True, which='both')
# plt.legend()





from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 5))

ax.set_xlim(x_h2[0], x_h2[-1])
ax.set_ylim(np.min(u_h2), np.max(u_h2))
ax.set_xlabel('x')
ax.set_ylabel('U')
ax.set_title('Эволюция приближенного решения (сетка h/2)')
ax.grid(True)

line, = ax.plot([], [], 'o-', markersize=2, linewidth=1.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def update(k):
    line.set_data(x_h2, u_h2[k, :])
    time_text.set_text(f't = {t_h2[k]:.4f}')
    return line, time_text


anim = FuncAnimation(
    fig,
    update,
    frames=len(t_h2),
    init_func=init,
    interval= 5,
    blit=True
)

plt.show()

# from mpl_toolkits.mplot3d import Axes3D


X, TT = np.meshgrid(x_h2, t_h2)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X,
    TT,
    u_h2,
    cmap='gnuplot',
    edgecolor='none',
    antialiased=True
)

ax.view_init(elev=30, azim=-120)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(t, x)')
ax.set_title('Поверхность приближенного решения U(t, x) на сетке h/2')

fig.colorbar(surf, ax=ax, shrink=0.8, pad=0.1, label='U(t, x)')

plt.show()