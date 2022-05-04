import tensorflow as tf

net = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
print(net(X))

print(net.layers[2].weights)
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))  # tf.Variable转tf.Tensor
print(net.get_weights())  # 获取所有参数，包括w和b，结果为数组




# 嵌套块参数
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
print(rgnet(X))

print(rgnet.summary())
print(rgnet.layers[0].layers[1].layers[1].weights[1])  # 这样访问嵌套块参数


# 参数初始化
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),  # 正态分布初始化
        bias_initializer=tf.zeros_initializer()),  # 0
    tf.keras.layers.Dense(1)])

net(X)
print(net.weights[0], net.weights[1])


net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),  # 初始化为常数1
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.weights[0], net.weights[1])


net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),  # xavier初始化
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

print(net(X))
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])



# 自定义初始化方法
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        """将绝对值<5的值设定为0"""
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

print(net(X))
print(net.layers[1].weights[0])

# 我们可以直接设置参数，因为为tf.Variable类型
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
print(net.layers[1].weights[0])


# 有时候我们想一个层多次出现
# tf.keras的表现有点不同。它会自动删除重复层，因此不能实现
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,  # 重复层
    tf.keras.layers.Dense(1),
])

print(net(X))
# 检查参数是否不同
print(net.summary())