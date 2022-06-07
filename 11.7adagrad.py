import math
import tensorflow as tf
from d2l import tensorflow as d2l

# 对于一个语言模型，有些词（特征）出现的很少，我们希望降低这些地方的学习率
# 一个解决办法是记录看见特征的次数s(i,t)（到t时的时候观察到特征i的次数）
# 瞬时学习率ηt = η0 / 根号下(s(i,t) + c)，c防止除零
# AdaGrad算法将计数器s(i,t)替换为先前观察到的梯度的平方和来解决这个问题
# 好处：我们获得了一个变化的梯度更新法，此外，我们能够平滑突然大或者小的梯度

# 梯度gt = l(yt, f(xt, w))对w的偏导数
# 状态st = st-1 + gt^2
# 更新权重wt = wt-1 - η / 根号下(st + ε) * gt

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

eta = 1
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
d2l.plt.show()


# 从零开始实现
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(tf.zeros((feature_dim, 1)))
    s_b = tf.Variable(tf.zeros(1))

def adagrad(params, grads, status, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, status, grads):
        s[:].assign(s + tf.math.square(g))  # 过去的s+现在的g
        p[:].assign(p - hyperparams['lr'] * g / # g/根号s
                    tf.math.sqrt(s + eps))

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim)
d2l.plt.show()


# 简洁实现
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {"learning_rate":0.1},
                       data_iter)
d2l.plt.show()