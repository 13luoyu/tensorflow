import tensorflow as tf
from d2l import tensorflow as d2l

# 考虑特殊的模型y = 0.1x1^2 + 2x2^2，梯度相差悬殊
eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    print(x1, x2)
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# 当前在x2方向梯度大，收敛快，但在x1方向收敛很慢
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()

# 提高lr，x2方向上发散，效果差
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
d2l.plt.show()
# 观察图像发现x2震荡，且逐渐增大，原因在于更新的时候，x2 = x1-eta*4*x2
# 而eta=0.6，相当于x2=-1.4x2，当然越来越大（学习率太大）

# 动量法可以解决上述问题
# 其实还是使用过去平衡了学习率过大的问题
# vt <- β*vt-1 + gt,t+1，其中β属于(0,1)，将瞬时梯度替换为过去梯度的均值，v被称为动量
# xt <- xt-1 - ηt*vt
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

# 降低动量比例
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

# 下面这个图说明了不同大小的动量比例对过去在将来结果中的影响力
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = tf.range(40).numpy()
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()
d2l.plt.show()





# 从零开始实现
def init_momentum_state(features_dim):
    v_w = tf.Variable(tf.zeros((features_dim, 1)))
    v_b = tf.Variable(tf.zeros(1))
    return (v_w, v_b)

def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
        v[:].assign(hyperparams['momentum'] * v + g)
        p[:].assign(p - hyperparams['lr'] * v)

def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_state(feature_dim),
                   {'lr':lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
d2l.plt.show()

# 增大动量，降低学习率，防止整体更新过大，步长不变，为(η/(1-β))
train_momentum(0.01, 0.9)
d2l.plt.show()


# 简洁实现
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
d2l.plt.show()


# 演示学习率lr对收敛的影响，考虑f(x)=1/2x^2, f'(x)=x
# xt+1 = xt - η * xt = (1-η)xt
# 这样，在t步之后，有xt = (1-η)^t * x0
eta = [0.01, 0.1, 1, 1.5, 2]  # 2以上无限发散
d2l.set_figsize((6,4))
for lr in eta:
    t = tf.range(20).numpy()
    d2l.plt.plot(t, (1-lr)**t, label=f'lr = {lr:.2f}')
d2l.plt.xlabel('t')
d2l.plt.ylabel('xt')
d2l.plt.legend()
d2l.plt.show()