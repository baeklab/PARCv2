import tensorflow as tf 
from parc.layers.regularizer import SpadeBlock
from parc.layers.resnet import ResNetBlock

class StyleMap(tf.keras.layers.Layer):
    def __init__(self, n_feats=128, n_out=1):
        super().__init__()
        self.style = SpadeBlock(n_feats)
        self.resnet = ResNetBlock(filters=n_feats, n_blocks=2, stride=1, n_out=n_feats)
        self.conv = tf.keras.layers.Conv2D(n_out, n_out, padding='same') # bug: output dimension

    def call(self, inputs):
        """
        Args: 
            inputs[0]:      couopled feature map
            inputs[1]:      advection and diffusion
        Returns:

        """
        x, mask = inputs[0], inputs[1]

        style_vector = self.style( [x, mask] )
        resnet_out = self.resnet(style_vector)
        conv_out = self.conv(resnet_out)
        return conv_out
