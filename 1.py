# 1. Даны значения величины заработной платы заемщиков банка(zp) и значения их поведенческого кредитного скоринга(ks):
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
# Используя математические операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла(то есть, ks - целевая переменная).
# Произвести расчет как с использованием intercept, так и без.
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


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

theta_init = np.zeros((1, 1))  # начальное значение коэффициентов
alpha = 0.0001  # скорость обучения
num_iters = 10000  # количество итераций

theta = gradient_descent(zp.reshape(
    (-1, 1)), ks.reshape((-1, 1)), theta_init, alpha, num_iters)
print('slope:', theta[0][0])


# Теперь посчитаем коэффициенты линейной регрессии с использованием intercept.

x_mean = np.mean(zp)
y_mean = np.mean(ks)

numerator = np.sum((zp - x_mean) * (ks - y_mean))
denominator = np.sum((zp - x_mean)**2)

beta1 = numerator / denominator
beta0 = y_mean - beta1 * x_mean

print('beta0:', beta0)
print('beta1:', beta1)

theta_init = np.array([[beta0], [beta1]])  # начальное значение коэффициентов
alpha = 0.0001  # скорость обучения
num_iters = 10000  # количество итераций

theta = gradient_descent(np.hstack((np.ones((len(zp), 1)), zp.reshape(
    (-1, 1)))), ks.reshape((-1, 1)), theta_init, alpha, num_iters)
print('intercept:', theta[0][0])
print('slope:', theta[1][0])
