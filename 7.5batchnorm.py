import tensorflow as tf
from d2l import tensorflow as d2l

# BN(x) = gamma * (x-mean)/var + beta
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # tf.math.rsqrt(X)返回X的平方根的倒数
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]  # 通道数
        self.gamma = self.add_weight(name="gamma", shape=weight_shape,
                                     initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name="beta", shape=weight_shape,
                                    initializer=tf.initializers.zeros, trainable=True)

        self.moving_mean = self.add_weight(name="moving_mean", shape=weight_shape,
                                           initializer=tf.initializers.zeros, trainable=False)
        self.moving_var = self.add_weight(name="moving_var", shape=weight_shape,
                                          initializer=tf.initializers.ones, trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            # 更新moving_mean和moving_var
            axis = list(range(len(inputs.shape) - 1))  # 除通道维的所有维度
            batch_mean = tf.reduce_mean(inputs, axis=axis, keepdims=True)
            batch_var = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)
            ), axis=axis, keepdims=True)
            # tf.squeeze()函数用于从张量形状中移除大小为1的维度
            batch_mean = tf.squeeze(batch_mean, axis=axis)
            batch_var = tf.squeeze(batch_var, axis=axis)
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            var_update = self.assign_moving_average(self.moving_var, batch_var)
            self.add_update(mean_update)
            self.add_update(var_update)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.moving_mean, self.moving_var
        output = batch_norm(inputs, self.gamma, self.beta, mean, var, eps=1e-5)
        return output

X = tf.random.normal((1,3,3,3))
layer = BatchNorm()
print(layer(X))



# 回想一下，这个函数必须传递给d2l.train_ch6。
# 或者说为了利用我们现有的CPU/GPU设备，需要在strategy.scope()建立模型
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )


lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()



def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()













