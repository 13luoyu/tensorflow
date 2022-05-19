import tensorflow as tf
from d2l import tensorflow as d2l

# 多头注意力有多个注意力，每个注意力头hi为
# hi = f(Wq*q, Wk*k, Wv*v)
# 之后连接后进入输出层，为
# Wo * [h1, ..., hn]

class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        # 输入queries, keys, values形状为(batch_size, 查询或键值对个数（二者相等）, num_hiddens)
        # valid_lens为(batch_size, ) or (batch_size, 查询个数)
        # 经过变换后，queries, keys, values形状为
        # (batch_size*num_heads, 查询或键值对个数, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # 在轴0，将第一项赋值num_heads次，后面同理
            valid_lens = tf.repeat(valid_lens, repeats = self.num_heads, axis=0)
        # output形状为(batch_size*num_heads, 查询或键值对个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    """为了多头注意力的并行计算而改变形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], num_heads, -1))
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = tf.transpose(X, perm=(0,2,1,3))
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

def transpose_output(X, num_heads):
    """transpose_qkv()函数的逆操作"""
    X = tf.reshape(X, shape=(-1, num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0,2,1,3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))






num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
print(attention(X, Y, Y, valid_lens, training=False).shape)





