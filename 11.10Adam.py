

# 在 11.4节中，我们学习了：随机梯度下降在解决优化问题时比梯度下降更有效。
#
# 在 11.5节中，我们学习了：在一个小批量中使用更大的观测值集，可以通过向量化提供额外效率。这是高效的多机、多GPU和整体并行处理的关键。
#
# 在 11.6节中我们添加了一种机制，用于汇总过去梯度的历史以加速收敛。
#
# 在 11.7节中，我们通过对每个坐标缩放来实现高效计算的预处理器。
#
# 在 11.8节中，我们通过学习率的调整来分离每个坐标的缩放。


# Adam算法将所有这些技术汇总到一个高效的学习算法中。
# 但是它并非没有问题，有时Adam算法可能由于方差控制不良而发散
#  在完善工作中，给Adam算法提供了一个称为Yogi的热补丁来解决这些问题

# Adam:
# 使用指数加权移动平均值来估算梯度的动量和二次矩，即它使用状态变量
# vt <- β1vt-1 + (1-β1)gt
# st <- β2st-1 + (1-β2)gt^2
# 通常，β1=0.9，β2=0.999， 也就是说，方差估计的移动远远慢于动量估计的移动
# 如此，如果我们初始化v0=s0=0，就会获得一个相当大的初始偏差，为此修正：
# vt^ = vt / (1-β1^t), st^ = st / (1-β2^t)
# gt' = η * vt^ / (根号下(st^) + ε)
# xt <- xt-1 - gt'


import tensorflow as tf
from d2l import tensorflow as d2l


def init_adam_states(feature_dim):
    v_w = tf.Variable(tf.zeros((feature_dim, 1)))
    v_b = tf.Variable(tf.zeros(1))
    s_w = tf.Variable(tf.zeros((feature_dim, 1)))
    s_b = tf.Variable(tf.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr
                    / tf.math.sqrt(s_bias_corr) + eps)


data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim)  # 简化了，t应该是1，2，3...
d2l.plt.show()

trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
d2l.plt.show()



# Adam算法也存在一些问题： 即使在凸环境下，当st的二次矩估计值爆炸时，它可能无法收敛（多轮迭代后st太大）
# 一个建议是重写st = βst-1 + (1-β)gt^2，即st = st-1 + (1-β)(gt^2 - st-1)为
# st = st-1 + (1-β)gt^2 * sgn(gt^2 - st-1)，sgn(x)=-1 if x<0, 0 if x==0 else 1
# 现在st不是递增，而是可以减少


def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad)) # 唯一不同
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim)
d2l.plt.show()