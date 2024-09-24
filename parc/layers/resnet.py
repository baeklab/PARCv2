import tensorflow as tf

class ResNetUnit(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, n_out=None):
        super().__init__()
        if n_out is not None:
            self.shortcut = tf.keras.layers.Conv2D(n_out, 1, strides=1, padding='same')
            self.c3 = tf.keras.layers.Conv2D(n_out, 1, strides=1, padding='same')
        else:
            self.shortcut = tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same')
            self.c3 = tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same')
        self.c1 = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', activation='relu')
        self.c2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', activation='relu')

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        added = tf.keras.layers.ReLU()(tf.math.add( shortcut, c3 ))
        return added

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, n_blocks, stride=1, n_out=None):
        super().__init__()
        self.n_blocks=n_blocks
        self.resnet_blocks = [None] * n_blocks
        for i in range(self.n_blocks - 1):
            self.resnet_blocks[i] = resnet_unit(filters, stride=stride)
        self.resnet_blocks[-1] = resnet_unit(filters, stride=stride, n_out=n_out) if n_out is not None else resnet_unit(filters, stride=stride)

    def call(self, inputs):
        x = self.resnet_blocks[0](inputs)
        for i in range(1, self.n_blocks):
            x = self.resnet_blocks[i](x)
        return x
