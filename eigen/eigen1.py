import numpy as np


f = open("eigen/text.txt")
n = int(f.readline())
A_list = []
for _ in range(n):
    A_list.append([float(i) for i in f.readline().split()])
f.close()

A = np.array(A_list, dtype=float)


def pm_power_method(A, y0=None, delta=1e-3, digits=6, max_iter=10000, min_iter=3):
    """
    PM-алгоритм (степенной метод) по Вержбицкому с пошаговой нормировкой.

    Важное уточнение для практики (следует из текста):
    - усреднять λ_i^(k) можно только когда они "устаканились" и почти одинаковы;
    - индексы с малыми |x_i^(k-1)| игнорируем (шаг 4).

    Параметры:
      delta   — допуск для шага 4 (в книге: "малое число δ>0")
      digits  — сколько десятичных знаков требуем "совпасть" на шаге 5
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be a square matrix")

    # Шаг 1
    if y0 is None:
        y0 = np.ones(n, dtype=float)
    else:
        y0 = np.array(y0, dtype=float).reshape(-1)
        if y0.size != n:
            raise ValueError("y0 must have size n")

    y0_norm = np.linalg.norm(y0)
    if y0_norm == 0:
        raise ValueError("Initial vector y0 must be non-zero")

    x_prev = y0 / y0_norm
    k = 1

    prev_mask = None
    lambdas_prev = None

    def rounded(v):
        return np.round(v, digits)

    # допуск на "почти одинаковы" внутри текущего набора λ_i^(k)
    spread_tol = 10.0 ** (-digits)

    for _ in range(max_iter):
        # Шаг 2
        y = A @ x_prev

        # Шаг 3
        y_norm = np.linalg.norm(y)
        if y_norm == 0:
            raise RuntimeError("Zero vector encountered: A x^(k-1) = 0")
        x = y / y_norm

        # Шаг 4
        mask = np.abs(x_prev) > delta
        if not np.any(mask):
            raise RuntimeError("All components of x^(k-1) are too small. Decrease delta or change y0.")

        lambdas = y[mask] / x_prev[mask]

        # Шаг 5 (как в тексте: игнорируем малые знаменатели; ждём, когда отношения устаканятся)
        if lambdas_prev is not None and prev_mask is not None and k >= min_iter:
            # сравниваем на пересечении допустимых индексов (маска могла измениться)
            common = mask & prev_mask
            if np.any(common):
                lam_cur_common = y[common] / x_prev[common]
                # lambdas_prev_common надо восстановить через прошлые значения:
                # проще хранить прошлый y_prev и x_prev_prev, но чтобы не усложнять —
                # проверяем стабильность по текущим/предыдущим rounded на пересечении
                # через сохранённые lambdas_prev в координатах prev_mask:
                # сделаем отображение индексов:
                idx_prev = np.where(prev_mask)[0]
                idx_map = {idx_prev[i]: i for i in range(idx_prev.size)}
                lam_prev_common = np.array([lambdas_prev[idx_map[i]] for i in np.where(common)[0]], dtype=float)

                stable_across_iters = np.all(rounded(lam_cur_common) == rounded(lam_prev_common))
            else:
                stable_across_iters = False

            # второе условие: внутри текущих lambdas они почти одинаковы (иначе усреднение рано)
            stable_within_set = (np.max(lambdas) - np.min(lambdas)) < spread_tol

            if stable_across_iters and stable_within_set:
                lam = float(np.mean(lambdas))
                return lam, x, k

        # обновления
        lambdas_prev = lambdas
        prev_mask = mask.copy()
        x_prev = x
        k += 1

    # если не сошлось — вернём последнюю оценку
    lam = float(np.mean(lambdas_prev)) if lambdas_prev is not None else None
    return lam, x_prev, k - 1


# ===== запуск =====
lam1, x1, iters = pm_power_method(A, y0=None, delta=1e-6, digits=6, max_iter=10000, min_iter=3)
print("lambda_1 ~", lam1)
print("iterations:", iters)
print("x_1 (normalized) =")
print(x1)
