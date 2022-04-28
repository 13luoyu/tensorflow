import tensorflow as tf

# 标量
x = tf.constant(3.0)
y = tf.constant(2.0)

print(x + y, x * y, x / y, x**y)


# 向量
x = tf.range(4)
print(x)
print(x[3])  # 元素
print(len(x))  # 长度
print(x.shape)  # 形状


# 矩阵
A = tf.reshape(tf.range(20), (5, 4))
print(A)
print(tf.transpose(A))  # 转置

# 张量，维度大于2的数据结构
X = tf.reshape(tf.range(24), (2, 3, 4))
print(X)



A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # A和B指向同一个对象
print(A, A + B, id(A), id(B))
print(A * B)  # 按元素点乘


# 降维
x = tf.range(4, dtype=tf.float32)
print(x, tf.reduce_sum(x))
# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。
# 我们还可以指定张量沿哪一个轴来通过求和降低维度。
print(A, tf.reduce_sum(A, axis=0))  # 同一列的所有值加到一列上，结果为一行
# 平均值
print(tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0])

# 保持维度的求和
print(tf.reduce_sum(A, axis=0, keepdims=True))


# 向量点积
x = tf.constant([1,2,3,4], dtype=tf.float32)
y = tf.ones(4)
print(tf.tensordot(x,y, axes=1))  # axes控制哪个变量进行转置，从而结果为标量、矩阵
print(tf.reduce_sum(x*y))  # 按元素点乘后求和，同样是点积

# 矩阵-向量积
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
print(tf.linalg.matvec(A, x))

# 矩阵乘法
B = tf.ones((4, 3), tf.float32)
print(tf.matmul(A, B))

# 范数
# L2范数，平方和再开方
u = tf.constant([3.0, -4.0])
print(tf.norm(u))
# L1范数，绝对值之和
print(tf.abs(u))
# 矩阵的Frobenius范数，矩阵元素平方和的平方根
u = tf.ones((4,9))
print(tf.norm(u))