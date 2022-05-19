import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l

# 自注意力，查询、键和值来自同一组输入，同一组词元同时充当查询、键和值
# yi = f(xi, (x1,x1), ..., (xn,xn))
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)

batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
print(attention(X, X, X, valid_lens, training=False).shape)


# 比较CNN、RNN和自注意力
# 目标：n个词元序列映射到另一个等长序列
# 总而言之，卷积神经网络和自注意力都拥有并行计算的优势， 而且自注意力的最大路径长度最短。
# 但是因为其计算复杂度是关于序列长度的二次方，所以在很长的序列中计算会非常慢。


# 位置编码
# 自注意力因为并行计算防疫了顺序操作，因此要再输入中添加位置编码表示每个词元的位置信息
# 输入为X(n,d)，n为词元数，d为词元表示维度，位置编码同为P(n,d)
# P(i, 2j) = sin(i / 10000^(2j/d)), P(i, 2j+1) = cos(i / 10000^(2j/d))

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # 足够长的P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(-1,1) / \
            np.power(10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:,:,0::2] = np.sin(X)
        self.P[:,:,1::2] = np.cos(X)
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
d2l.plt.show()


P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
d2l.plt.show()