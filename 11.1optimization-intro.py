import numpy as np
import tensorflow as tf
from mpl_toolkits import mplot3d
from d2l import tensorflow as d2l

# 优化函数的目标是基于损失函数，减小训练误差，而深度学习的目标是减小泛化误差
# 因此，除了减少损失函数训练误差外，还要防止过拟合

# 风险函数f：衡量整个数据群的预期损失
def f(x):
    return x * tf.cos(np.pi * x)
# 经验风险函数g：训练数据集的平均损失
def g(x):
    return f(x) + 0.2 * tf.cos(5 * np.pi * x)

def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = tf.range(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
d2l.plt.show()

# 在本章中，我们将特别关注优化算法在最小化目标函数方面的性能，而不是模型的泛化误差。
# 在 3.1节中，我们区分了优化问题中的解析解和数值解。在深度学习中，大多数目标函数都很复杂，没有解析解。
# 相反，我们必须使用数值优化算法。

# 局部最小值
x = tf.range(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
d2l.plt.show()
# 深度学习模型的目标函数通常有许多局部最优解。当优化问题的数值解接近局部最优值时，随着目标函数解的梯度接近或变为零，
# 通过最终迭代获得的数值解可能仅使目标函数局部最优，而不是全局最优。只有一定程度的噪声可能会使参数超出局部最小值。
# 事实上，这是小批量随机梯度下降的有利特性之一，在这种情况下，小批量上梯度的自然变化能够将参数从局部极小值中移出。

# 鞍点
x = tf.range(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
d2l.plt.show()
# 三维鞍点
#  [A,B]=tf.meshgrid(a,b) 将a重复b行,得到A 将b重复a列,得到B 最后大家的size都是 b*a
x, y = tf.meshgrid(
    tf.linspace(-1.0, 1.0, 101), tf.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.show()

# 我们假设函数的输入是k维向量，其输出是标量，因此其Hessian矩阵（二阶导数矩阵）将有k特征值
# 函数的解决方案可以是局部最小值、局部最大值或函数梯度为零的位置处的鞍点：
#
# 当函数在零梯度位置处的Hessian矩阵的特征值全部为正值时，我们有该函数的局部最小值。
#
# 当函数在零梯度位置处的Hessian矩阵的特征值全部为负值时，我们有该函数的局部最大值。
#
# 当函数在零梯度位置处的Hessian矩阵的特征值为负值和正值时，我们对函数有一个鞍点。

# 梯度消失
x = tf.range(-2.0, 5.0, 0.01)
d2l.plot(x, [tf.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
d2l.plt.show()
# 解决：引入ReLU函数
