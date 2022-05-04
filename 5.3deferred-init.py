# 延后初始化
# 我们定义了网络架构，但没有指定输入维度。
# 我们添加层时没有指定前一层的输出维度。
# 我们在初始化参数时，甚至没有足够的信息来确定模型应该包含多少参数。

# 框架的延后初始化（defers initialization）， 即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。


import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

print([net.layers[i].get_weights() for i in range(len(net.layers))])

X = tf.random.uniform((2, 20))
print(net(X))
print([w.shape for w in net.get_weights()])

