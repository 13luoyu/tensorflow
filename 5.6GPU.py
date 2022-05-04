import tensorflow as tf

print(tf.device('/CPU:0'), tf.device('/GPU:0'))
# 查询可用gpu数量
print(len(tf.config.experimental.list_physical_devices('GPU')))

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

print(try_gpu(), try_gpu(10), try_all_gpus())



x = tf.constant([1, 2, 3])
print(x.device)  # 直接就在GPU上了

with tf.device("/CPU:0"):
    x = tf.ones((2,3))  # 在CPU上创建张量
print(x.device)

with try_gpu():
    x = tf.ones((2,3))  # 在GPU上创建张量
print(x.device)


# 在GPU上创建网络
# tf.distribute.MirroredStrategy 支持在单机多GPU上的同步分布式训练. 它在每个GPU设备上创建一个副本. 模型中的每个变量都将在所有副本之间进行镜像。
# 这些变量一起形成一个称为MirroredVariable的概念上的变量。通过应用相同的更新，这些变量彼此保持同步。
# 高效的归约算法用于在设备之间传递变量更新。全归约通过对不同设备上的张量相加进行聚合, 并使他们在所有设备上可用。
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
net(x)
print(net.layers[0].weights[0].device)


# 默认也是在GPU上创建
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1)
])
net(x)
print(net.layers[0].weights[0].device)