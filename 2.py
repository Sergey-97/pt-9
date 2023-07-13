# 2. Посчитать коэффициент линейной регрессии при заработной плате(zp), используя градиентный спуск(без intercept).

import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient_descent(X, y, theta_init, alpha, num_iters):
    m = len(y)
    theta = theta_init

    for i in range(num_iters):
        y_pred = np.dot(X, theta)
        error = y_pred - y
        grad = (1/m) * np.dot(X.T, error)
        theta -= alpha * grad

    return theta

# данные
zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

# гиперпараметры
theta_init = np.zeros((1, 1))
alpha = 0.0001
num_iters = 10000

# обучение модели
theta = gradient_descent(zp.reshape(
    (-1, 1)), ks.reshape((-1, 1)), theta_init, alpha, num_iters)

# вывод результата
print('slope:', theta[0][0])
