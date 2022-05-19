

# RNN中，本步的输入为上一步的隐状态和输入，本部输出为输出和本步隐状态
# Ht = f(XtWxh + Ht-1Whh + Bh)
# 隐状态中XtWxh + Ht-1Whh的计算， 相当于Xt和Ht-1的拼接与Wxh和Whh的拼接的矩阵乘法

import tensorflow as tf
from d2l import tensorflow as d2l

X, W_xh = tf.random.normal((3, 1), 0, 1), tf.random.normal((1, 4), 0, 1)
H, W_hh = tf.random.normal((3, 4), 0, 1), tf.random.normal((4, 4), 0, 1)
print(tf.matmul(X, W_xh) + tf.matmul(H, W_hh))

print(tf.matmul(tf.concat((X, H), 1), tf.concat((W_xh, W_hh), 0)))

