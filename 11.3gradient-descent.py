import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

# 梯度下降：
# 由一阶泰勒公式：f(x+ε)=f(x)+εf'(x)+O(ε^2)
# 我们假设f'(x)>0，即沿着负梯度方向移动会减少f，考虑固定步长η>0，令ε=-ηf'(x)，则
# f(x-ηf'(x)) = f(x) - ηf'(x)^2 + O(η^2f'(x)^2)
# f(x-ηf'(x)) <= f(x)，因此更新可以是
# x <- x - η*f'(x)
def f(x):
    return x ** 2
def f_grad(x):
    return x * 2

def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    return results
results = gd(0.2, f_grad)
print(results)

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = tf.range(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]],
             'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
d2l.plt.show()
# 小学习率
show_trace(gd(0.05, f_grad), f)
d2l.plt.show()
# 大学习率
show_trace(gd(1.1, f_grad), f)
d2l.plt.show()

# 局部最小值
c = tf.constant(0.15 * np.pi)
def f(x):
    return x * tf.cos(c * x)
def f_grad(x):
    return tf.cos(c * x) - c * x * tf.sin(c * x)
show_trace(gd(2, f_grad), f)
d2l.plt.show()



# 多元梯度下降
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """用定制的训练机优化2D目标函数"""
    # s1和s2是稍后将使用的内部状态变量
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    return results

def show_trace_2d(f, results):  #@save
    """显示优化过程中2D变量的轨迹"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = tf.meshgrid(tf.range(-5.5, 1.0, 0.1),
                          tf.range(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # 目标函数的梯度
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
d2l.plt.show()






# 牛顿法，将二阶梯度也考虑进来
# f(x+ε) = f(x) + εf'(x) + 1/2εTf''(x)ε + O(ε^3)，令H为f''(x)的Hessian矩阵
# 对ε求导，有f'(x) + εH = 0，所以可以取ε = -f'(x)H^-1
# 例f(x) = 1/2x^2, f'(x)=x, H=1，则ε=-x，一步收敛，原因在于f(x)的泰勒展开是准确的
# 下面考虑不准确的情况

# 凸函数
c = tf.constant(0.5)
def f(x):  # O目标函数
    return tf.cosh(c * x)
def f_grad(x):  # 目标函数的梯度
    return c * tf.sinh(c * x)
def f_hess(x):  # 目标函数的Hessian
    return c**2 * tf.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    return results

show_trace(newton(), f)
d2l.plt.show()

# 非凸函数
c = tf.constant(0.15 * np.pi)
def f(x):  # 目标函数
    return x * tf.cos(c * x)
def f_grad(x):  # 目标函数的梯度
    return tf.cos(c * x) - c * x * tf.sin(c * x)
def f_hess(x):  # 目标函数的Hessian
    return - 2 * c * tf.sin(c * x) - x * c**2 * tf.cos(c * x)

show_trace(newton(), f)
d2l.plt.show()

# 上面没有在最小值收敛，调参
show_trace(newton(0.5), f)
d2l.plt.show()




