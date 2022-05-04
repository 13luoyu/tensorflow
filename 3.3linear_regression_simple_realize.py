import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个TensorFlow数据迭代器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# keras是TensorFlow的高级API
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))  # 1为输出值数

# 初始化模型参数
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
# 上面的代码可能看起来很简单，但是你应该注意到这里的一个细节：
# 我们正在为网络初始化参数，而Keras还不知道输入将有多少维!
# 网络的输入可能有2维，也可能有2000维。 Keras让我们避免了这个问题，
# 在后端执行时，初始化实际上是推迟（deferred）执行的。
# 只有在我们第一次尝试通过网络传递数据时才会进行真正的初始化。
# 请注意，因为参数还没有初始化，所以我们不能访问或操作它们。

# 损失函数
loss = tf.keras.losses.MeanSquaredError()
# 优化算法
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(y, net(X, training=True))
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net.get_weights()[0]
print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b的估计误差：', true_b - b)