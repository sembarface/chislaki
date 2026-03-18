import numpy as np
import matplotlib.pyplot as plt


# Краевая задача
# y'' - p(x)*y = f(x),   x in [a,b]
#
# y'(a) - alpha1*y(a) = alpha2
# y'(b) - beta1*y(b)  = beta2


def solve_bvp_d(p, f, a, b, alpha1, alpha2, beta1, beta2, h, use_bc6=False):

    q = (b - a) / h
    N = int(round(q))

    if abs(q - N) > 1e-12:
        raise ValueError("(b - a) должно делиться на h")

    if N < 7:
        raise ValueError("Нужно достаточно мелкое h, чтобы N >= 7")

    x = a + h * np.arange(N + 1)

    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    # ЛЕВОЕ ГРАНИЧНОЕ УСЛОВИЕ
    if not use_bc6:
        # O(h^3)
        # y'(a) = (-11 y0 + 18 y1 - 9 y2 + 2 y3)/(6h) + O(h^3)
        A[0, 0] = -11.0/(6.0*h) - alpha1
        A[0, 1] =  18.0/(6.0*h)
        A[0, 2] =  -9.0/(6.0*h)
        A[0, 3] =   2.0/(6.0*h)
        rhs[0] = alpha2
    else:
        # O(h^6)
        # y'(a) =
        # (-147 y0 + 360 y1 - 450 y2 + 400 y3 - 225 y4 + 72 y5 - 10 y6)/(60 h)
        # + O(h^6)
        A[0, 0] = -147.0/(60.0*h) - alpha1
        A[0, 1] =  360.0/(60.0*h)
        A[0, 2] = -450.0/(60.0*h)
        A[0, 3] =  400.0/(60.0*h)
        A[0, 4] = -225.0/(60.0*h)
        A[0, 5] =   72.0/(60.0*h)
        A[0, 6] =  -10.0/(60.0*h)
        rhs[0] = alpha2

    # =====================================================
    # УЗЕЛ i = 1 : y'' = O(h^6)
    # =====================================================
    i = 1
    A[i, 0] =   7.0/10.0   / h**2
    A[i, 1] = - 7.0/18.0   / h**2 - p(x[i])
    A[i, 2] = -27.0/10.0   / h**2
    A[i, 3] =  19.0/4.0    / h**2
    A[i, 4] = -67.0/18.0   / h**2
    A[i, 5] =   9.0/5.0    / h**2
    A[i, 6] = - 1.0/2.0    / h**2
    A[i, 7] =  11.0/180.0  / h**2
    rhs[i] = f(x[i])

    # =====================================================
    # УЗЕЛ i = 2 : y'' = O(h^6)
    # =====================================================
    i = 2
    A[i, 0] = -11.0/180.0  / h**2
    A[i, 1] = 107.0/90.0   / h**2
    A[i, 2] = -21.0/10.0   / h**2 - p(x[i])
    A[i, 3] =  13.0/18.0   / h**2
    A[i, 4] =  17.0/36.0   / h**2
    A[i, 5] = - 3.0/10.0   / h**2
    A[i, 6] =   4.0/45.0   / h**2
    A[i, 7] = - 1.0/90.0   / h**2
    rhs[i] = f(x[i])

    # =====================================================
    # ЦЕНТРАЛЬНЫЕ УЗЛЫ: i = 3, ..., N-3
    # 7-точечная симметричная формула O(h^6)
    # =====================================================
    for i in range(3, N - 2):
        A[i, i-3] =   1.0/90.0   / h**2
        A[i, i-2] = - 3.0/20.0   / h**2
        A[i, i-1] =   3.0/2.0    / h**2
        A[i, i]   = -49.0/18.0   / h**2 - p(x[i])
        A[i, i+1] =   3.0/2.0    / h**2
        A[i, i+2] = - 3.0/20.0   / h**2
        A[i, i+3] =   1.0/90.0   / h**2
        rhs[i] = f(x[i])

    # =====================================================
    # УЗЕЛ i = N-2 : y'' = O(h^6)
    # =====================================================
    i = N - 2
    A[i, N-7] = - 1.0/90.0   / h**2
    A[i, N-6] =   4.0/45.0   / h**2
    A[i, N-5] = - 3.0/10.0   / h**2
    A[i, N-4] =  17.0/36.0   / h**2
    A[i, N-3] =  13.0/18.0   / h**2
    A[i, N-2] = -21.0/10.0   / h**2 - p(x[i])
    A[i, N-1] = 107.0/90.0   / h**2
    A[i, N]   = -11.0/180.0  / h**2
    rhs[i] = f(x[i])

    # =====================================================
    # УЗЕЛ i = N-1 : y'' = O(h^6)
    # =====================================================
    i = N - 1
    A[i, N-7] =  11.0/180.0  / h**2
    A[i, N-6] = - 1.0/2.0    / h**2
    A[i, N-5] =   9.0/5.0    / h**2
    A[i, N-4] = -67.0/18.0   / h**2
    A[i, N-3] =  19.0/4.0    / h**2
    A[i, N-2] = -27.0/10.0   / h**2
    A[i, N-1] = - 7.0/18.0   / h**2 - p(x[i])
    A[i, N]   =   7.0/10.0   / h**2
    rhs[i] = f(x[i])

    # =====================================================
    # ПРАВОЕ ГРАНИЧНОЕ УСЛОВИЕ
    # =====================================================
    if not use_bc6:
        # O(h^3)
        # y'(b) = (11 yN - 18 yN-1 + 9 yN-2 - 2 yN-3)/(6h) + O(h^3)
        A[N, N]   =  11.0/(6.0*h) - beta1
        A[N, N-1] = -18.0/(6.0*h)
        A[N, N-2] =   9.0/(6.0*h)
        A[N, N-3] =  -2.0/(6.0*h)
        rhs[N] = beta2
    else:
        # O(h^6)
        # y'(b) =
        # (147 yN - 360 yN-1 + 450 yN-2 - 400 yN-3 + 225 yN-4 - 72 yN-5 + 10 yN-6)/(60 h)
        # + O(h^6)
        A[N, N]   =  147.0/(60.0*h) - beta1
        A[N, N-1] = -360.0/(60.0*h)
        A[N, N-2] =  450.0/(60.0*h)
        A[N, N-3] = -400.0/(60.0*h)
        A[N, N-4] =  225.0/(60.0*h)
        A[N, N-5] =  -72.0/(60.0*h)
        A[N, N-6] =   10.0/(60.0*h)
        rhs[N] = beta2

    y = np.linalg.solve(A, rhs)

    return x, y


# Ошибки

def errors_on_grid(x, y, y_exact):
    u = y_exact(x)
    e = y - u
    err_max = np.max(np.abs(e))
    err_l2 = np.sqrt(np.mean(e * e))
    return u, e, err_max, err_l2


# Тестовая задача с точным решением

def y_exact(x):
    return np.exp(x) + x**2

def p(x):
    return 1.0 + x

def f(x):
    return (np.exp(x) + 2.0) - (1.0 + x) * (np.exp(x) + x**2)


a = 0.0
b = 1.0

alpha1 = 1.0
beta1 = -1.0

alpha2 = (np.exp(a) + 2.0*a) - alpha1 * (np.exp(a) + a*a)
beta2  = (np.exp(b) + 2.0*b) - beta1  * (np.exp(b) + b*b)



use_bc6 = False


# Решение на двух сетках
h1 = 0.1
h2 = h1 / 2.0

x1, y1 = solve_bvp_d(p, f, a, b, alpha1, alpha2, beta1, beta2, h1, use_bc6=use_bc6)
x2, y2 = solve_bvp_d(p, f, a, b, alpha1, alpha2, beta1, beta2, h2, use_bc6=use_bc6)

u1, e1, err1, l2_1 = errors_on_grid(x1, y1, y_exact)
u2, e2, err2, l2_2 = errors_on_grid(x2, y2, y_exact)

p_max = np.log2(err1 / err2)
p_l2 = np.log2(l2_1 / l2_2)

print("=== Режим границ ===")
if use_bc6:
    print("Граничные условия: O(h^6)")
else:
    print("Граничные условия: O(h^3)")

print("\n=== Ошибки ===")
print(f"h   = {h1:.6f}   max|e| = {err1:.6e}   L2 = {l2_1:.6e}")
print(f"h/2 = {h2:.6f}   max|e| = {err2:.6e}   L2 = {l2_2:.6e}")

print("\n=== Фактический порядок ===")
print("p_max =", p_max)
print("p_L2  =", p_l2)

plt.figure()
plt.plot(x2, u2, label="exact")
plt.plot(x1, y1, "o-", label="h")
plt.plot(x2, y2, ".-", label="h/2")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()

plt.figure()
plt.plot(x1, np.abs(e1), "o-", label="error h")
plt.plot(x2, np.abs(e2), ".-", label="error h/2")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("error")
plt.grid(True, which="both")
plt.legend()

plt.show()