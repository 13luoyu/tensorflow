import random
import tensorflow as tf
from d2l import tensorflow as d2l



def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = tf.zeros((num_examples, w.shape[0]))
    # 前面说过，Tensor不可赋值，这里成立的原因在于，两个tensor相加得到一个tensor，使用X引用它
    X += tf.random.normal(shape=X.shape)  # 正态分布
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].numpy(), labels.numpy(), 1)
d2l.plt.show()




# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
# tf.gather(params, indices, validate_indices=None, name=None, axis=0)
# 就是抽取出params的第axis维度上在indices索引里面所有的值
# 之所以这里不用features[j]，因为[]运算符只支持
# integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)  # x为batch_size*2，y为batch_size*1
    break


# 初始化模型参数
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

def linreg(X, w, b):  #定义模型
    """线性回归模型"""
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):  #损失函数
    """均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, grads, lr, batch_size):  #优化算法
    """小批量随机梯度下降"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)  # 这里除以batch_size，因为grad计算了所有批量的和
# tf.assign_sub(ref, value)，使用ref减去value

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 计算l关于[w,b]的梯度
        dw, db = g.gradient(l, [w, b])
        # 使用参数的梯度更新参数
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

print(f'w的估计误差: {true_w - tf.reshape(w, true_w.shape)}')
print(f'b的估计误差: {true_b - b}')