import tensorflow as tf
from parc.layers.resnet import ResNetBlock

class Resnet27(tf.keras.layers.Layer):
    def __init__(self, n_out=128):
        super().__init__()
        self.block1 = ResNetBlock(filters=64, n_blocks=2, stride=1)
        self.block2 = ResNetBlock(filters=128, n_blocks=2, stride=1)
        self.block3 = ResNetBlock(filters=256, n_blocks=3, stride=1)
        self.block4 = ResNetBlock(filters=512, n_blocks=2, stride=1, n_out=n_out) 
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out
    
class Resnet50(tf.keras.layers.Layer):
    def __init__(self, n_out=128):
        super().__init__()
        self.block1 = ResNetBlock(filters=64, n_blocks=3, stride=1)
        self.block2 = ResNetBlock(filters=128, n_blocks=4, stride=1)
        self.block3 = ResNetBlock(filters=256, n_blocks=6, stride=1)
        self.block4 = ResNetBlock(filters=512, n_blocks=3, stride=1, n_out=n_out) 
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out

class Resnet101(tf.keras.layers.Layer):
    def __init__(self, n_out=128):
        super().__init__()
        self.block1 = ResNetBlock(filters=64, n_blocks=3, stride=1)
        self.block2 = ResNetBlock(filters=128, n_blocks=4, stride=1)
        self.block3 = ResNetBlock(filters=256, n_blocks=23, stride=1)
        self.block4 = ResNetBlock(filters=512, n_blocks=3, stride=1, n_out=n_out)
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out

class Resnet152(tf.keras.layers.Layer):
    def __init__(self, n_out=128):
        super().__init__()
        self.block1 = ResNetBlock(filters=64, n_blocks=3, stride=1)
        self.block2 = ResNetBlock(filters=128, n_blocks=8, stride=1)
        self.block3 = ResNetBlock(filters=256, n_blocks=36, stride=1)
        self.block4 = ResNetBlock(filters=512, n_blocks=3, stride=1, n_out=n_out)
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out
