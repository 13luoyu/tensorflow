import tensorflow as tf
# 自动微分

x = tf.range(4, dtype=tf.float32)

x = tf.Variable(x)

# 把所有计算记录在磁带上
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)  # y=2x^2
print(y)

#  接下来，我们通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
x_grad = t.gradient(y, x)  # 导数y=4x
print(x_grad)

with tf.GradientTape() as t:
    y = tf.reduce_sum(x)  # y=x
print(t.gradient(y, x))


# 设置persistent=True来运行t.gradient多次
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)  # 这里意为下面的操作中，将u看作常数，不计算梯度
    z = u * x

x_grad = t.gradient(z, x)  #
print(x_grad == u)
print(t.gradient(y, x) == 2*x)



# 使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），
# 我们仍然可以计算得到的变量的梯度。

def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

a = tf.Variable(tf.random.normal(shape=()))  # a为一个标量
with tf.GradientTape() as t:
    d = f(a)  # d = ka
d_grad = t.gradient(d, a)
print(d_grad == d/a)
