import tensorflow as tf
from d2l import tensorflow as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))

# 交叉熵损失函数
# Sparse_categorical_crossentropy函数中的参数from_logits用法：
# 其中形参默认为from_logits=False，网络预测值y_pred 表示必须为经过了 Softmax函数的输出值。
# 当 from_logits = True 时，网络预测值y_pred 表示必须为还没经过 Softmax 函数的变量
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 小批量随机梯度下降算法
trainer = tf.keras.optimizers.SGD(learning_rate=.1)  # .1就是0.1

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()

d2l.predict_ch3(net, test_iter)
d2l.plt.show()