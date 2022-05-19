import tensorflow as tf
from d2l import tensorflow as d2l


def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y

X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias


X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
print(X)
K = tf.constant([[1.0, -1.0]])
Y = corr2d(X, K)  # 这种卷积核可以检测X的这种垂直边缘
print(Y)

print(corr2d(tf.transpose(X), K))  # 转置就不能检测了




# 训练
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = tf.keras.layers.Conv2D(1, (1,2), use_bias=False)
X = tf.reshape(X, (1,6,8,1))  # 批量数，高，宽，通道数
Y = tf.reshape(Y, (1,6,7,1))
lr = 3e-2
Y_hat = conv2d(X)

for i in range(10):
    # GradientTape默认只监控由tf.Variable创建的traiable=True属性（默认）的变量的梯度。
    # 下面先设置不监控，然后手动指定监控conv2d.weights[0]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        update = tape.gradient(l, conv2d.weights[0]) * lr
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i+1) % 2 == 0:
            print(f'epoch {i+1}, loss {tf.reduce_sum(l):.3f}')
