import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['STXihei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

x = np.arange(-5, 5)
y = 2 * x + 5

X_blue = np.array([-3., -4., 1., -2.5])
Y_blue = np.array([2., -1., 10.5, 3.])

X_red = np.array([2.5, 3., 1., 4.])
Y_red = np.array([4., 6., -2., -1.2])

plt.title("线性可分的一个例子")

plt.xlabel("x 轴")
plt.ylabel("y 轴")
plt.plot(x, y)

plt.scatter(X_blue, Y_blue, edgecolors='b')
plt.scatter(X_red, Y_red, edgecolors='r')

plt.show()
