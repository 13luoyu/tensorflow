import tensorflow as tf

x = tf.range(12)
print(x)  # tf.Tensor([ 0  1  2  3  4  5  6  7  8  9 10 11], shape=(12,), dtype=int32)
print(x.shape)  # (12,)
print(tf.size(x))  # tf.Tensor(12, shape=(), dtype=int32)，x中元素数

x = tf.reshape(x, (3,4))
print(x)  # tf.Tensor(
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]], shape=(3, 4), dtype=int32)

print(tf.zeros((2,3,4)))
x = tf.ones((2,3,4))

# 均值0、方差1的正态分布
x = tf.random.normal(shape=[3,4])
print(x)

# 从python列表创建tensor
x = tf.constant([[1,2],[3,4]])
print(x)


# 运算符
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
print(x+y, x-y, x*y, x/y, x**y)  # **为幂运算，都是对应位置元素进行运算

print(tf.exp(x))  # 指数函数

x = tf.reshape(tf.range(12, dtype=tf.float32), (3,4))
y = tf.constant([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(tf.concat([x,y], axis=0))  # 行增加的连结
print(tf.concat([x,y], axis=1))  # 列增加的连结

print(x == y)

# 对所有元素求和
print(tf.reduce_sum(x))  # tf.Tensor(66.0, shape=(), dtype=float32)

# 广播机制
a = tf.reshape(tf.range(3), (3,1))
b = tf.reshape(tf.range(2), (1,2))
print(a, b, a+b)

# 索引和切片
print(x[-1], x[1:3])
# Tensorflow中Tensor值是不可变的，不能被赋值。Tensorflow的Variables是支持赋值的容器
x_var = tf.Variable(x)
x_var[1,2].assign(100)  # 这样赋值
print(x_var)
# 为多个元素赋值
x_var[0:2, :].assign(tf.ones(x_var[0:2, :].shape, dtype=tf.float32) * 12)
print(x_var)


# 转换为其他python对象
# 与numpy的转换
a = x.numpy()
b = tf.constant(a)
print(type(a), type(b))










