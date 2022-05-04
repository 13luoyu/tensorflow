import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)

# 保存模型两种都可以
net.save("mlp")
tf.saved_model.save(net, "mlp")
# 加载模型
net = tf.saved_model.load("mlp")

Y_clone = net(X)
print(Y_clone == Y)


net = MLP()
Y = net(X)
# 保存模型参数
net.save_weights("mlp.params")

net.load_weights("mlp.params")
Y_clone = net(X)
print(Y_clone == Y)