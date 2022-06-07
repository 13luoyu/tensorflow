import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

# 使用矩阵点乘演示效率
# 1、我们可以通过点积进行逐元素计算。
# 2、我们可以一次计算一列或一行
# 3、我们可以简单地计算A=BC
# 4、我们可以将B和C分成较小的区块矩阵，然后一次计算A的一个区块

timer = d2l.Timer()
A = tf.Variable(tf.zeros((256, 256)))
B = tf.Variable(tf.random.normal([256, 256], 0, 1))
C = tf.Variable(tf.random.normal([256, 256], 0, 1))


# 逐元素计算A=BC，执行太慢注释掉了
timer.start()
# for i in range(256):
#     for j in range(256):
#         A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
print(timer.stop())

timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
print(timer.stop())

# 一次性计算A=BC
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
print(timer.stop())

# 虽然3性能高，但有时可能无法将矩阵全部读入内存，因此4诞生了

# 批量法
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
print(timer.stop())



#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),  # 使用前1500个数据
                               batch_size, is_train=True)  # 每次返回batch_size个xy
    return data_iter, data.shape[1]-1


# minbatch-sgd从0开始实现
def sgd(params, grads, states, hyperparams):
    """要更新的参数，梯度，状态，超参数"""
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr'] * grad)


#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # 训练模型
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

# 批量梯度下降，总共1500个batch，1次迭代完成
gd_res = train_sgd(1, 1500, 10)
d2l.plt.show()
# 随机梯度下降，总共1个batch，需要1500次迭代
sgd_res = train_sgd(0.005, 1)
d2l.plt.show()
# 小批量随机梯度下降，100个batch，15次迭代
mini1_res = train_sgd(.4, 100)
d2l.plt.show()
# 小批量随机梯度下降，10个batch，150次迭代，因为迭代次数多，所以需要时间更长
mini2_res = train_sgd(.05, 10)
d2l.plt.show()

# 统计时间和loss
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
d2l.plt.show()








# 简洁实现
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # 初始化模型
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # MeanSquaredError计算平方误差时不带系数1/2
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')

data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
d2l.plt.show()