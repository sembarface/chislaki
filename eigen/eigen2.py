import numpy as np


f = open("eigen/text.txt")
n = int(f.readline())
A_list = []
for _ in range(n):
    A_list.append([float(i) for i in f.readline().split()])
f.close()

A = np.array(A_list, dtype=float)


def jacobi_rotations(A, eps=1e-10, max_iter=1_000_000):
    """
    Метод вращений Якоби (классический: ключевой элемент max |a_ij|, i<j)
    для симметричной матрицы A.

    Останов: max_{i<j} |b_ij| < eps
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("Jacobi method requires a symmetric matrix")

    B = A.copy()
    V = np.eye(n, dtype=float)  # накопление собственных векторов

    it = 0
    while it < max_iter:
        # --- максимум над диагональю ---
        abs_upper = np.abs(np.triu(B, 1))
        max_off = abs_upper.max()

        # критерий останова (ВАЖНО: до argmax)
        if max_off < eps:
            break

        i, j = np.unravel_index(np.argmax(abs_upper), abs_upper.shape)
        aij = B[i, j]
        aii = B[i, i]
        ajj = B[j, j]

        # --- шаг из Вержбицкого: p, q, d ---
        p = 2.0 * aij
        q = aii - ajj
        d = np.sqrt(p * p + q * q)
        if d == 0.0:
            break

        # --- вычисление c, s ---
        if q != 0.0:
            r = abs(q) / (2.0 * d)
            c = np.sqrt(0.5 + r)
            s = np.sqrt(0.5 - r)
            sign_pq = 1.0 if (p * q) >= 0.0 else -1.0
            s *= sign_pq

            # стабилизация из текста для |p| << |q|
            if abs(p) < 1e-8 * abs(q):
                s = abs(p) * sign_pq / (2.0 * c * d)
        else:
            c = np.sqrt(0.5)
            s = np.sqrt(0.5)

        # --- новые диагональные элементы ---
        bii = c*c*aii + s*s*ajj + 2.0*c*s*aij
        bjj = s*s*aii + c*c*ajj - 2.0*c*s*aij
        B[i, i] = bii
        B[j, j] = bjj

        # зануляем b_ij
        B[i, j] = 0.0
        B[j, i] = 0.0

        # --- пересчёт строк/столбцов i и j (формула (4.31)) ---
        for m in range(n):
            if m == i or m == j:
                continue
            bim = B[i, m]
            bjm = B[j, m]

            new_bim = c * bim + s * bjm
            new_bjm = -s * bim + c * bjm

            B[i, m] = new_bim
            B[m, i] = new_bim
            B[j, m] = new_bjm
            B[m, j] = new_bjm

        # --- накопление V <- V T ---
        vi = V[:, i].copy()
        vj = V[:, j].copy()
        V[:, i] = c * vi + s * vj
        V[:, j] = -s * vi + c * vj

        it += 1

    eigenvalues = np.diag(B).copy()
    eigenvectors = V

    # сортировка (по убыванию)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, it


# ===== запуск =====
vals, vecs, iters = jacobi_rotations(A, eps=1e-12)

print("iterations:", iters)
print("eigenvalues:")
print(vals)
#print("eigenvectors (columns):")
#print(vecs)


w, v = np.linalg.eigh(A)
print("numpy.eigh eigenvalues:", w[::-1])
