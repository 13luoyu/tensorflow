import tensorflow as tf
from d2l import tensorflow as d2l

# 使用注意力实现seq2seq
# key和value都是encoder的输出，query是encoder的最后一层state，以及decoder每一层state

#@save
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(key_size=num_hiddens, query_size=num_hiddens,
                                               num_hiddens=num_hiddens, dropout=dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout) for _ in range(num_layers)]
        ), return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(num_steps，batch_size，num_hiddens)
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state, enc_valid_lens)

    def call(self, X, state, **kwargs):
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X形状为(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        X = tf.transpose(X, perm=(1,0,2))  # (num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X:  # 对每个时间步, x: (batch_size, embed_size)
            # query为(batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # content为(batch_size, 1, num_hiddens)
            # key和value都是encoder的输出，query是encoder的最后一层state，以及decoder每一层state
            content = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
            # 在特征维度相连
            x = tf.concat((content, tf.expand_dims(x, axis=1)), axis=-1)  # batch_size, 1, num_hiddens+embed_size
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
X = tf.zeros((4, 7))  # (batch_size, num_steps)
encoder_outputs = encoder(X, training=False)
state = decoder.init_state(encoder_outputs, None)
output, state = decoder(X, state, training=False)
# 分别是decoder output，3state内容, encoder output(不变), hidden_state, hidden_state第1层0时间步的形状
print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)


# 训练
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
d2l.plt.show()


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

attention_weights = tf.reshape(
    tf.concat([step[0][0][0] for step in dec_attention_weight_seq], 0),
    (1, 1, -1, num_steps))
# 加上一个包含序列结束词元
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')
d2l.plt.show()
















