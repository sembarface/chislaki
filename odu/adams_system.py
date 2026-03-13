import numpy as np
import matplotlib.pyplot as plt



# ===== Пример 1 — простая устойчивая система =====
# гармонический осциллятор
# u1' = u2
# u2' = -u1

def f(t, u):
    return np.array([
        u[1],
        -u[0]
    ])

def u_exact(t):
    t = np.asarray(t)
    return np.column_stack((
        np.cos(t),
        -np.sin(t)
    ))

t0 = 0.0
T  = 10.0
u0 = u_exact(np.array([t0]))[0]
h = 0.2


# ===== Пример 2 — нелинейная система =====

# u1 = e^t
# u2 = e^(2t)

# def f(t, u):
#     return np.array([
#         u[0] + u[1] - np.exp(2*t),
#         2*u[1]
#     ])

# def u_exact(t):
#     t = np.asarray(t)
#     return np.column_stack((
#         np.exp(t),
#         np.exp(2*t)
#     ))

# t0 = 0.0
# T  = 3.0
# u0 = u_exact(np.array([t0]))[0]
# h = 0.2


# ===== Пример 3 — жесткая система =====
# u1' = -50u1 + sin(t)
# u2' = -u2

# def f(t, u):
#     return np.array([
#         -50*u[0] + np.sin(t),
#         -u[1]
#     ])

# def u_exact(t):
#     t = np.asarray(t)
#     u1 = (50*np.sin(t) - np.cos(t) + np.exp(-50*t)) / 2501
#     u2 = np.exp(-t)
#     return np.column_stack((u1, u2))

# t0 = 1.0
# T  = 2.0
# u0 = u_exact(np.array([t0]))[0]
# h = 0.2


# RK6
def rk6_step(f, t, y, h):
    k1 = f(t, y)

    k2 = f(t + h/3.0,
           y + h*(k1/3.0))

    k3 = f(t + 2.0*h/3.0,
           y + h*(2.0*k2/3.0))

    k4 = f(t + h/3.0,
           y + h*(k1/12.0 + k2/3.0 - k3/12.0))

    k5 = f(t + h/2.0,
           y + h*(-k1/16.0 + 9.0*k2/8.0 - 3.0*k3/16.0 - 3.0*k4/8.0))

    k6 = f(t + h/2.0,
           y + h*(9.0*k2/8.0 - 3.0*k3/8.0 - 3.0*k4/4.0 + k5/2.0))

    k7 = f(t + h,
           y + h*(9.0*k1/44.0 - 9.0*k2/11.0 + 63.0*k3/44.0 + 18.0*k4/11.0 - 16.0*k6/11.0))

    return y + h*(11.0*k1/120.0 + 27.0*k3/40.0 + 27.0*k4/40.0 - 4.0*k5/15.0 - 4.0*k6/15.0 + 11.0*k7/120.0)


# 5-шаговый Adams-Moulton
def solve_adams_moulton_5step(f, t0, T, y0, h, it_max=50, tol=1e-12):

    N = int(np.round((T - t0) / h))
    t = t0 + h*np.arange(N + 1)

    y0 = np.asarray(y0, dtype=float)
    m = y0.size

    y = np.zeros((N + 1, m), dtype=float)
    y[0] = y0

    # старт RK6
    for n in range(1, min(5, N + 1)):
        y[n] = rk6_step(f, t[n-1], y[n-1], h)

    if N < 4:
        return t, y

    # значения f
    fn = np.zeros((N + 1, m), dtype=float)
    for n in range(0, min(5, N + 1)):
        fn[n] = f(t[n], y[n])

    # основной цикл
    for n in range(5, N + 1):

        f1 = fn[n-1]
        f2 = fn[n-2]
        f3 = fn[n-3]
        f4 = fn[n-4]

        # предиктор AB4
        y_pred = y[n-1] + (h/24.0)*(55*f1 - 59*f2 + 37*f3 - 9*f4)

        const = 646*f1 - 264*f2 + 106*f3 - 19*f4

        # итерации
        y_s = y_pred.copy()

        for _ in range(it_max):
            y_next = y[n-1] + (h/720.0)*(251*f(t[n], y_s) + const)

            if np.linalg.norm(y_next - y_s) < tol*(1 + np.linalg.norm(y_next)):
                y_s = y_next
                break

            y_s = y_next

        y[n] = y_s
        fn[n] = f(t[n], y[n])

    return t, y


# ошибки
def errors_on_grid(t, y, u_exact):
    u = u_exact(t)
    e = y - u

    err_max = np.max(np.linalg.norm(e, axis=1))
    err_l2 = np.sqrt(np.mean(np.linalg.norm(e, axis=1)**2))

    return u, e, err_max, err_l2


# решения
t_h, y_h = solve_adams_moulton_5step(f, t0, T, u0, h)
t_h2, y_h2 = solve_adams_moulton_5step(f, t0, T, u0, h/2)


# ошибки
u_h, e_h, emax_h, el2_h = errors_on_grid(t_h, y_h, u_exact)
u_h2, e_h2, emax_h2, el2_h2 = errors_on_grid(t_h2, y_h2, u_exact)


# порядок
p_max = np.log2(emax_h / emax_h2)
p_l2 = np.log2(el2_h / el2_h2)

print("=== Ошибки ===")
print(f"h   = {h}   max|e| = {emax_h:.6e}   L2 = {el2_h:.6e}")
print(f"h/2 = {h/2} max|e| = {emax_h2:.6e}   L2 = {el2_h2:.6e}")

print("\n=== Фактический порядок ===")
print("p_max =", p_max)
print("p_L2  =", p_l2)


# графики решений
m = y_h.shape[1]

plt.figure(figsize=(8, 5))
for i in range(m):
    plt.plot(t_h2, u_h2[:, i], label=f"exact u{i+1}")
    plt.plot(t_h,  y_h[:, i], "o-", label=f"num u{i+1} (h)")
    plt.plot(t_h2, y_h2[:, i], ".-", label=f"num u{i+1} (h/2)")

plt.xlabel("t")
plt.ylabel("solution components")
plt.title("Numerical vs exact solutions")
plt.grid(True)
plt.legend()


# графики ошибок
plt.figure(figsize=(8, 5))
for i in range(m):
    plt.plot(t_h,  np.abs(e_h[:, i]), "o-", label=f"|error u{i+1}| (h)")
    plt.plot(t_h2, np.abs(e_h2[:, i]), ".-", label=f"|error u{i+1}| (h/2)")

plt.yscale("log")
plt.xlabel("t")
plt.ylabel("error")
plt.title("Errors of solution components")
plt.grid(True, which="both")
plt.legend()


# норма ошибки всей системы
plt.figure(figsize=(8, 5))
plt.plot(t_h,  np.linalg.norm(e_h, axis=1), "o-", label="||error|| (h)")
plt.plot(t_h2, np.linalg.norm(e_h2, axis=1), ".-", label="||error|| (h/2)")

plt.yscale("log")
plt.xlabel("t")
plt.ylabel("||error||")
plt.title("Norm of system error")
plt.grid(True, which="both")
plt.legend()

plt.show()