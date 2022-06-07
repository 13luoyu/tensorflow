

# Adadelta:
# Adadelta没有学习率参数。相反，它使用参数本身的变化率来调整学习率。
#
# Adadelta需要两个状态变量来存储梯度和参数的变化。
#
# Adadelta使用泄漏的平均值来保持对适当统计数据的运行估计。

# 牛顿法中考虑二阶导（11.3），有参数的更新量Δx = f'(x)/f''(x)
# 这里借鉴，有1 / f''(x) = Δx / f'(x)，因此应当将学习率变为一个Δx，采用平方的指数衰减移动均方根（RMS）
# adagrad方法中，更新为xt+1 = xt - η / 根号下(st+ε) * gt
# 将学习率换为RMS即可

# st = ρst-1 + (1-ρ)gt^2
# g = 根号下(Δxt-1 + ε) / 根号下(st + ε) * gt  g其实就是参数变化量
# xt = xt-1 - g
# Δxt = ρΔxt-1 + (1-ρ)g^2



import tensorflow as tf
from d2l import tensorflow as d2l


def init_adadelta_states(feature_dim):
    s_w = tf.Variable(tf.zeros((feature_dim, 1)))
    s_b = tf.Variable(tf.zeros(1))
    delta_w = tf.Variable(tf.zeros((feature_dim, 1)))
    delta_b = tf.Variable(tf.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)


data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim)
d2l.plt.show()


# adadelta is not converging at default learning rate
# but it's converging at lr=5.0
# 怪了，tensorflow的adadelta只在lr=5.0时收敛
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate': 5.0,'rho': 0.9}, data_iter)
d2l.plt.show()