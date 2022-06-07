import math
import tensorflow as tf
from d2l import tensorflow as d2l

# Adagrad算法将梯度gt的评分累加成状态，即st = st-1 + gt^2
# 在 AdaGrad 算法中，由于梯度分量的直接累加，
# 步长随着迭代的进行而单调递减， 这可能导致后期步长过小无法收敛。
# 解决问题一种方法是使用均值st/t，问题在于这样将流程每个值看作等权重
# 不妥。因此RMSProp采用的方法和动量法类似，使用泄露平均值
# st <- γst-1 + (1-γ)gt^2，其他部分不变

# st <- γst-1 + (1-γ)gt^2
# xt <- xt-1 - η / 根号下(st + ε) * gt

# st = (1-γ)gt^2 + γst-1
#    = (1-γ)(gt^2 + γgt-1^2 + ...)
# 因此使用1 + γ + ... = 1 / (1-γ)
# 权重总和标准化为1且观测值的半衰期为1/γ


# 绘图，经过t步后d0在st中剩下的部分
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = tf.range(40).numpy()
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time')
d2l.plt.show()


def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2*x1, 4*x2, 1e-6
    s1 = gamma * s1 + (1-gamma) * g1 ** 2
    s2 = gamma * s2 + (1-gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
d2l.plt.show()







def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(tf.zeros((feature_dim, 1)))
    s_b = tf.Variable(tf.zeros(1))
    return (s_w, s_b)

def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))


data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim)
d2l.plt.show()



trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
d2l.plt.show()