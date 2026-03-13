import numpy as np
import matplotlib.pyplot as plt



# ===== Пример 1 — простое устойчивое уравнение =====
# u' = -u ,  u(0) = 1

def f(t, u):
    return -u

def u_exact(t):
    return np.exp(-t)

t0 = 0.0
T  = 5.0
u0 = u_exact(t0)
h = 0.5


# ===== Пример 2 — нелинейное уравнение =====

# def f(t, u):
#     return ((u+t)*np.log((u+t)/t) + u)/(t)

# def u_exact(t):
#     return (t*np.exp(t)-t)

# t0 = 1.0
# T  = 5.0
# u0 = u_exact(t0)
# h = 0.5


# ===== Пример 3 — жесткое уравнение =====
# u' = -50u + sin(t) , u(0)=0

# def f(t, u):
#     return -50*u + np.sin(t)

# def u_exact(t):
#     return (50*np.sin(t) - np.cos(t) + np.exp(-50*t)) / 2501

# t0 = 1.0
# T  = 2.0
# u0 = u_exact(t0)
# h = 0.5




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


# 5-шаговый Adams–Moulton

def solve_adams_moulton_5step(f, t0, T, y0, h, it_max=50, tol=1e-12):

    N = int(np.round((T - t0) / h))
    t = t0 + h*np.arange(N + 1)

    y = np.zeros(N + 1)
    y[0] = y0

    for n in range(1, min(5, N + 1)):
        y[n] = rk6_step(f, t[n-1], y[n-1], h)

    if N < 4:
        return t, y

    fn = np.zeros(N + 1)

    for n in range(0, 5):
        fn[n] = f(t[n], y[n])

    for n in range(5, N + 1):

        f1 = fn[n-1]
        f2 = fn[n-2]
        f3 = fn[n-3]
        f4 = fn[n-4]

        y_pred = y[n-1] + (h/24.0)*(55*f1 - 59*f2 + 37*f3 - 9*f4)

        const = (646*f1 - 264*f2 + 106*f3 - 19*f4)

        y_s = y_pred

        for _ in range(it_max):

            y_next = y[n-1] + (h/720.0)*(251*f(t[n], y_s) + const)

            if abs(y_next - y_s) < tol*(1 + abs(y_next)):
                y_s = y_next
                break

            y_s = y_next

        y[n] = y_s
        fn[n] = f(t[n], y[n])

    return t, y


# ошибки

def errors_on_grid(t, y, u_exact):

    if u_exact is None:
        return None, None, None, None

    u = u_exact(t)
    e = y - u

    err_max = np.max(np.abs(e))
    err_l2  = np.sqrt(np.mean(e*e))

    return u, e, err_max, err_l2


# решение

t_h, y_h = solve_adams_moulton_5step(f, t0, T, u0, h)
t_h2, y_h2 = solve_adams_moulton_5step(f, t0, T, u0, h/2)

u_h, e_h, emax_h, el2_h = errors_on_grid(t_h, y_h, u_exact)
u_h2, e_h2, emax_h2, el2_h2 = errors_on_grid(t_h2, y_h2, u_exact)

# графики

plt.figure()

if u_exact is not None:
    plt.plot(t_h2, u_h2, label="exact")

plt.plot(t_h, y_h, "o-", label="h")
plt.plot(t_h2, y_h2, ".-", label="h/2")

plt.xlabel("t")
plt.ylabel("solution")
plt.grid()
plt.legend()


# погрешности

p_max = np.log2(emax_h / emax_h2)
p_l2 = np.log2(el2_h / el2_h2)

print("=== Ошибки ===")
print(f"h   = {h}   max|e| = {emax_h:.6e}   L2 = {el2_h:.6e}")
print(f"h/2 = {h/2} max|e| = {emax_h2:.6e}   L2 = {el2_h2:.6e}")

print("\n=== Фактический порядок ===")
print("p_max =", p_max)
print("p_L2  =", p_l2)

plt.figure()

plt.plot(t_h, np.abs(e_h), "o-", label="error h")
plt.plot(t_h2, np.abs(e_h2), ".-", label="error h/2")

plt.yscale("log")
plt.xlabel("t")
plt.ylabel("error")
plt.grid(True, which="both")
plt.legend()


plt.show()