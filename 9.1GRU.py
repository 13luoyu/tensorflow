import tensorflow as tf
from d2l import tensorflow as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 初始化模型参数
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    def three():
        return (tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
                tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params

def init_gru_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])  # (batch_size, num_inputs)
        # 更新门结果
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        # 重置门结果
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        # 候选隐状态
        H_delta = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H * R, W_hh) + b_h)
        # 隐状态
        H = Z * H + (1 - Z) * H_delta
        # 输出
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)

vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
num_epochs, lr = 500, 1
with strategy.scope():
    net = d2l.RNNModelScratch(vocab_size, num_hiddens, init_gru_state,
                              gru, get_params)
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
d2l.plt.show()






# 简洁实现
gru_cell = tf.keras.layers.GRUCell(num_hiddens, kernel_initializer="glorot_uniform")
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True, return_state=True, return_sequences=True)
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = d2l.RNNModel(gru_layer, vocab_size)
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
d2l.plt.show()