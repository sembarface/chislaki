import numpy as np
import matplotlib.pyplot as plt


def f(t, u):
    # пример: u' = -u, u(0)=1
    return -u

def u_exact(t):
    # точное решение для примера: exp(-t)
    return np.exp(-t)

t0 = 0.0
T  = 5.0
u0 = 1.0

h = 0.5  # базовый шаг



def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h,     y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def solve_adams_moulton_5step(f, t0, T, y0, h, it_max=50, tol=1e-12):
    """
    5-шаговый неявный метод Адамса (Adams–Moulton, порядок 5):
      y_n = y_{n-1} + h/720*(251 f_n + 646 f_{n-1} -264 f_{n-2} +106 f_{n-3} -19 f_{n-4})

    Неявность решаем простыми итерациями:
      y^{s+1} = y_{n-1} + h/720*(251 f(t_n, y^s) + остальное)

    Начальное приближение y^(0) берём предиктором Adams–Bashforth 4-го порядка:
      y_pred = y_{n-1} + h/24*(55 f_{n-1} -59 f_{n-2} +37 f_{n-3} -9 f_{n-4})
    """
    N = int(np.round((T - t0) / h))
    t = t0 + h*np.arange(N + 1)

    y = np.zeros(N + 1, dtype=float)
    y[0] = y0

    # старт: нужно y1..y4 -> RK4
    for n in range(1, min(5, N + 1)):
        y[n] = rk4_step(f, t[n-1], y[n-1], h)

    if N < 4:
        return t, y

    # заранее посчитаем f-values
    fn = np.zeros(N + 1, dtype=float)
    for n in range(0, 5):
        fn[n] = f(t[n], y[n])

    # основной цикл: n = 5..N
    for n in range(5, N + 1):
        # значения прошлых f
        f1 = fn[n-1]
        f2 = fn[n-2]
        f3 = fn[n-3]
        f4 = fn[n-4]

        # предиктор AB4
        y_pred = y[n-1] + (h/24.0) * (55.0*f1 - 59.0*f2 + 37.0*f3 - 9.0*f4)

        # константная часть для корректора
        const = (646.0*f1 - 264.0*f2 + 106.0*f3 - 19.0*f4)

        # итерации для неявного шага
        y_s = y_pred
        for _ in range(it_max):
            y_next = y[n-1] + (h/720.0) * (251.0*f(t[n], y_s) + const)
            if abs(y_next - y_s) < tol * (1.0 + abs(y_next)):
                y_s = y_next
                break
            y_s = y_next

        y[n] = y_s
        fn[n] = f(t[n], y[n])

    return t, y


def errors_on_grid(t, y, u_exact):
    u = u_exact(t)
    e = y - u
    err_max = np.max(np.abs(e))
    err_l2  = np.sqrt(np.mean(e*e))
    return u, e, err_max, err_l2


# ====== Решение с шагом h и h/2 ======
t_h,  y_h  = solve_adams_moulton_5step(f, t0, T, u0, h)
t_h2, y_h2 = solve_adams_moulton_5step(f, t0, T, u0, h/2)

# сравнение с точным
u_h,  e_h,  emax_h,  el2_h  = errors_on_grid(t_h,  y_h,  u_exact)
u_h2, e_h2, emax_h2, el2_h2 = errors_on_grid(t_h2, y_h2, u_exact)

# оценка фактического порядка (по max и по L2)
p_max = np.log2(emax_h / emax_h2) if (emax_h2 > 0) else np.inf
p_l2  = np.log2(el2_h  / el2_h2)  if (el2_h2  > 0) else np.inf

print("=== Ошибки относительно точного решения ===")
print(f"h   = {h: .6g}  max|e| = {emax_h: .6e}   L2 = {el2_h: .6e}")
print(f"h/2 = {h/2: .6g}  max|e| = {emax_h2: .6e}   L2 = {el2_h2: .6e}")
print()
print("=== Фактический порядок ===")
print(f"p_fact (max) = {p_max: .4f}")
print(f"p_fact (L2)  = {p_l2: .4f}")

# ====== Графики ======
plt.figure()
plt.plot(t_h2, u_h2, label="exact")
plt.plot(t_h,  y_h,  "o-", label=f"Adams-Moulton 5-step, h={h}")
plt.plot(t_h2, y_h2, ".-", label=f"Adams-Moulton 5-step, h={h/2}")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t_h,  np.abs(e_h),  "o-", label=f"|error|, h={h}")
plt.plot(t_h2, np.abs(e_h2), ".-", label=f"|error|, h={h/2}")
plt.yscale("log")
plt.xlabel("t")
plt.ylabel("|u_num - u_exact|")
plt.grid(True, which="both")
plt.legend()


plt.show()
