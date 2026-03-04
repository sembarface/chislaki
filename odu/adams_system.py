import numpy as np
import matplotlib.pyplot as plt


# ===== пример системы =====
# u1' = u2
# u2' = -u1

def f(t, u):
    return np.array([
        u[1],
        -u[0]
    ])


# точное решение для примера
def u_exact(t):
    return np.vstack([
        np.cos(t),
        -np.sin(t)
    ]).T


t0 = 0.0
T = 10.0
u0 = np.array([1.0, 0.0])

h = 0.2


# ===== RK4 шаг для системы =====
def rk4_step(f, t, y, h):

    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)

    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# ===== 5-шаговый Adams-Moulton =====
def solve_adams_moulton_5step(f, t0, T, y0, h, it_max=50, tol=1e-12):

    N = int(np.round((T - t0) / h))
    t = t0 + h*np.arange(N + 1)

    m = len(y0)

    y = np.zeros((N + 1, m))
    y[0] = y0

    # старт RK4
    for n in range(1, min(5, N + 1)):
        y[n] = rk4_step(f, t[n-1], y[n-1], h)

    if N < 4:
        return t, y

    # значения f
    fn = np.zeros((N + 1, m))
    for n in range(0, 5):
        fn[n] = f(t[n], y[n])

    # основной цикл
    for n in range(5, N + 1):

        f1 = fn[n-1]
        f2 = fn[n-2]
        f3 = fn[n-3]
        f4 = fn[n-4]

        # предиктор AB4
        y_pred = y[n-1] + (h/24.0)*(55*f1 - 59*f2 + 37*f3 - 9*f4)

        const = (646*f1 - 264*f2 + 106*f3 - 19*f4)

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


# ===== ошибки =====
def errors_on_grid(t, y, u_exact):

    u = u_exact(t)

    e = y - u

    err_max = np.max(np.linalg.norm(e, axis=1))
    err_l2 = np.sqrt(np.mean(np.linalg.norm(e, axis=1)**2))

    return u, e, err_max, err_l2


# ===== решения =====
t_h, y_h = solve_adams_moulton_5step(f, t0, T, u0, h)
t_h2, y_h2 = solve_adams_moulton_5step(f, t0, T, u0, h/2)


# ===== ошибки =====
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


# ===== графики решений =====

m = y_h.shape[1]

plt.figure(figsize=(8,5))

for i in range(m):

    plt.plot(t_h2, u_h2[:, i], label=f"exact u{i+1}")
    plt.plot(t_h,  y_h[:, i], "o-", label=f"num u{i+1} (h)")
    plt.plot(t_h2, y_h2[:, i], ".-", label=f"num u{i+1} (h/2)")

plt.xlabel("t")
plt.ylabel("solution components")
plt.title("Numerical vs exact solutions")
plt.grid(True)
plt.legend()


# ===== графики ошибок =====

plt.figure(figsize=(8,5))

for i in range(m):

    plt.plot(t_h,  np.abs(e_h[:, i]), "o-", label=f"|error u{i+1}| (h)")
    plt.plot(t_h2, np.abs(e_h2[:, i]), ".-", label=f"|error u{i+1}| (h/2)")

plt.yscale("log")
plt.xlabel("t")
plt.ylabel("error")
plt.title("Errors of solution components")
plt.grid(True, which="both")
plt.legend()



# ===== норма ошибки всей системы =====

plt.figure(figsize=(8,5))

plt.plot(t_h,  np.linalg.norm(e_h, axis=1), "o-", label="||error|| (h)")
plt.plot(t_h2, np.linalg.norm(e_h2, axis=1), ".-", label="||error|| (h/2)")

plt.yscale("log")
plt.xlabel("t")
plt.ylabel("||error||")
plt.title("Norm of system error")
plt.grid(True, which="both")
plt.legend()
plt.show()