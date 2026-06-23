from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)  # x data points
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)  # y data points


def best_fit_slope_and_intercept(xs, ys):
    # slope m using the least-squares formula
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    # intercept b from slope and means
    b = mean(ys) - m * mean(xs)
    return m, b


def squared_error(ys_orig, ys_line):
    # sum of squared differences between two y-sets
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]          # flat line at mean(y)
    squared_error_regr = squared_error(ys_orig, ys_line)     # error of regression line
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)  # error of mean line
    return 1 - (squared_error_regr / squared_error_y_mean)   # R^2 value


m, b = best_fit_slope_and_intercept(xs, ys)  # compute slope & intercept

regression_line = [(m * x) + b for x in xs]  # y-values predicted by the line

predict_x = 8  # new x to predict
predict_y = (m * predict_x) + b  # predicted y for predict_x

r_squared = coefficient_of_determination(ys, regression_line)  # goodness of fit
print(r_squared)

plt.scatter(xs, ys)                          # plot original data
plt.scatter(predict_x, predict_y, color='g')  # plot predicted point
plt.plot(xs, regression_line)                # plot best-fit line
plt.show()


