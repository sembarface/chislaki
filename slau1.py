f = open("text.txt")
n = int(f.readline())
A = []
for _ in range(n):
    A.append([float(i) for i in f.readline().split()])
b = [float(i) for i in f.readline().split()]
f.close()


def print_matrix(A):
    for i in range(n):
        for j in range(n):
            print(f"{A[i][j]:10.6f}", end=" ")
        print()
    print()


def gauss(A, b, eps=1e-10):
    A_copy = [row[:] for row in A]
    b_copy = b[:]

    # Прямой ход метода Гаусса
    for k in range(n - 1):
        max_row = k
        max_val = abs(A_copy[k][k])

        for i in range(k + 1, n):
            if abs(A_copy[i][k]) > max_val:
                max_val = abs(A_copy[i][k])
                max_row = i

        if max_row != k:
            A_copy[k], A_copy[max_row] = A_copy[max_row], A_copy[k]
            b_copy[k], b_copy[max_row] = b_copy[max_row], b_copy[k]

        for i in range(k + 1, n):
            if abs(A[k][k])<=eps:continue #Проверка на нулевой столбец
            t = A_copy[i][k] / A_copy[k][k]

            for j in range(k + 1, n):
                A_copy[i][j] -= t * A_copy[k][j]

            b_copy[i] -= t * b_copy[k]

            A_copy[i][k] = 0.0

    # Обратный ход метода Гаусса
    x = [0.0] * n
    if abs(A_copy[n-1][n-1])<=eps:
        return 0 #Матрица вырождена

    for i in range(n - 1, -1, -1):

        s = 0.0
        for j in range(i + 1, n):
            s += A_copy[i][j] * x[j]

        x[i] = (b_copy[i] - s) / A_copy[i][i]

    return x


print("Исходная матрица A:")
print_matrix(A)
print(f"Вектор b: {b}")

solution = gauss(A, b)
if (solution!=0):
    print("Решение системы:")
    for i, x_val in enumerate(solution):
        print(f"x{i + 1} = {x_val:.8f}")
else:
    print("Матрица вырождена")

'''
3
3 2 -5
2 -1 3
1 2 -1
-1 13 9
'''
