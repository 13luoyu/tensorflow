import math
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from d2l import tensorflow as d2l


def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 代码几乎与d2l.train_ch6定义在卷积神经网络一章LeNet一节中的相同
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net


# 默认设置
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
d2l.plt.show()



# 更通用，我们定义一个调度器，当调用更新次数时，返回新的学习率
# 例如η = η0 * (t+1)^(-1/2)
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr
    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)

scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
d2l.plt.show()


# 应用在训练
train(net, train_iter, test_iter, num_epochs, lr, custom_callback=LearningRateScheduler(scheduler))
d2l.plt.show()
# 这比以前好一些：曲线比以前更加平滑，并且过拟合更小了。



# 下面介绍一些调度器
# 多项式衰减——乘法衰减，每次乘以一个小值，ηt+1 = max(ηmin, ηt*α)
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr
    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(tf.range(50), [scheduler(t) for t in range(50)])
d2l.plt.show()

# 每隔一段时间再衰减，在指定的step进行乘法衰减
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
d2l.plt.show()

train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
d2l.plt.show()


# 余弦调度器
# 思想：我们可能不想在一开始就太大地降低学习率，而且可能希望最终能用非常小的学习率来“改进”解决方案
# ηt = ηT + (η0-ηT)/2 * (1 + cos(Πt/T))，其中η0为初始学习率，ηT为T时的目标学习率，对于
# t>T，固定值为ηT而不再增加
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update  # T epoch后不再更新
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:  # 预热
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:  # 权重衰退
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
d2l.plt.show()

train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
d2l.plt.show()

# 注意到在余弦调度器中有个预热，
# 在某些情况下，初始化参数不足以得到良好的解。 这对于某些高级网络设计来说尤其棘手，
# 可能导致不稳定的优化结果。 对此，一方面，我们可以选择一个足够小的学习率， 从而防止一开始发散，
# 然而这样进展太缓慢。 另一方面，较高的学习率最初就会导致发散。
#
# 解决这种困境的一个相当简单的解决方法是使用预热期，在此期间学习率将增加至初始最大值。
# 预热期结束后后冷却直到优化过程结束。 为了简单起见，通常使用线性递增。 这引出了如下表所示的时间表。
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
d2l.plt.show()

train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
d2l.plt.show()
# 注意，观察前5个迭代轮数的性能，网络最初收敛得更好。