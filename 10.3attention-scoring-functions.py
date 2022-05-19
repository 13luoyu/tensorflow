import tensorflow as tf
from d2l import tensorflow as d2l

def masked_softmax(X, valid_len):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_len is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_len.shape) == 1:
            valid_len = tf.repeat(valid_len, repeats=shape[1])
        else:
            valid_len = tf.reshape(valid_len, shape=-1)
        X = d2l.sequence_mask(tf.reshape(X, shape=(-1, shape[-1])),
                              valid_len, value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)

print(masked_softmax(tf.random.uniform(shape=(2,2,4)), tf.constant([2,3])))
print(masked_softmax(tf.random.uniform(shape=(2,2,4)), tf.constant([[1,3],[2,4]])))


# 前面的高斯核其实就是评分函数a(q, k) = softmax((q-ki)**2/2)
# 加性注意力评分函数a(q, k) = Wv * tanh(Wq*q + Wk*k)
class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 维度扩展后，queries为(batch_size, 查询数, 1, num_hiddens)
        # keys为(batch_size, 1, 键值对数, num_hiddens)
        # 之后广播机制求和
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(keys, axis=1)
        features = tf.tanh(features)
        # scores形状(batch_size, 查询数, 键值对数)
        scores = tf.squeeze(self.W_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values形状: (batch_size, 键值对数, 值的维度)
        # 返回维度(batch_size, 查询数, 值的维度)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

queries, keys = tf.random.normal(shape=(2,1,20)), tf.ones((2,10,2))
values = tf.repeat(tf.reshape(tf.range(40, dtype=tf.float32), shape=(1,10,4)),
                   repeats=2, axis=0)
valid_lens = tf.constant([2,4])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
print(attention(queries, keys, values, valid_lens, training=False))

d2l.show_heatmaps(tf.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()



# 缩放点积注意力
# a(q, k) = qk/根号d，其中d为k, q的长度，要求两者长度相同
#@save
class DotProductAttention(tf.keras.layers.Layer):
    """Scaleddotproductattention."""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def call(self, queries, keys, values, valid_lens, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)

queries = tf.random.normal(shape=(2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention(queries, keys, values, valid_lens, training=False)

d2l.show_heatmaps(tf.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()