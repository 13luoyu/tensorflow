import tensorflow as tf
from d2l import tensorflow as d2l

d2l.use_svg_display()
# fashion_mnist图片分类数据集
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
print(len(mnist_train[0]), len(mnist_test[0]))
print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):  #@save
    """根据label的index返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

images = tf.constant(mnist_train[0][:18])
labels = tf.constant(mnist_train[1][:18])
show_images(images,2,9,titles=get_fashion_mnist_labels(labels))
d2l.plt.show()

batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(mnist_train)\
    .batch(batch_size).shuffle(len(mnist_train[0]))







def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # 将所有图像X除以255，使所有像素值介于0和1之间，在最后添加一个批处理维度，
    # 并将标签y转换为int32。
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (  # 如果resize不为None，则改变图像大小
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break