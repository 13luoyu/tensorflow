import tensorflow as tf
from d2l import tensorflow as d2l


def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])

X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

class TrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        super().__init__()
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                     legend=['train_loss', 'train_acc', 'test_acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name

    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()

    def on_epoch_end(self, epoch, logs=None):
        self.timer.stop()
        test_acc = self.net.evaluate(self.test_iter, verbose=0, return_dict=True)['accuracy']
        # def evaluate(self,
        #                x=None,
        #                y=None,
        #                batch_size=None,
        #                verbose=1,
        #                sample_weight=None,
        #                steps=None,
        #                callbacks=None,
        #                max_queue_size=10,
        #                workers=1,
        #                use_multiprocessing=False,
        #                return_dict=False):
        # verbose, 0 or 1,0不在标准输出流输出日志信息，1输出，默认为1
        # return_dict: If `True`, loss and metric results are returned as a dict,
        #           with each key being the name of the metric. If `False`, they are
        #           returned as a list.
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch+1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]  # 第一个[0]指的是[数据, 标签]中的数据
            # tf.data.experimental.cardinality()确定在数据集中可用的数据批次，即60000/batch+size向上取整
            num_examples = batch_size * tf.data.experimental.cardinality(self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')


def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # 'accuracy'表示y和y_hat都是数值
    callback = TrainCallback(net, train_iter, test_iter, num_epochs, device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    # def fit(self,
    #           x=None,
    #           y=None,
    #           batch_size=None,
    #           epochs=1,
    #           verbose=1,  # 日志显示，0表示不在标准输出打印日志信息，1表示输出，2表示每个epoch输出一行
    #           callbacks=None,
    #           validation_split=0.,  # 用来指定一定比例的数据作为验证集
    #           validation_data=None,  # 指定验证集（x，y）
    #           shuffle=True,
    #           ...)
    return net




lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()