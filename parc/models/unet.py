import tensorflow as tf

class Unet(tf.keras.layers.Layer):
    def __init__(self, n_feats=64, n_feature_map=128):
        """ Implementation of the UNet model (ref: https://arxiv.org/pdf/1505.04597) with modified skip connections
        Args: 
            n_feats:        (int)   base number of features for UNet construction
            n_feature_map:  (int)   number of feature map for output
        """
        super().__init__()
        self.d1c1 = tf.keras.layers.Conv2D(n_feats, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.d1c2 = tf.keras.layers.Conv2D(n_feats, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same')

        self.d2c0 = tf.keras.layers.MaxPooling2D(2,2)
        self.d2c1 = tf.keras.layers.Conv2D(n_feats * 2, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.d2c2 = tf.keras.layers.Conv2D(n_feats * 2, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.d3c0 = tf.keras.layers.MaxPooling2D(2,2)
        self.d3c1 = tf.keras.layers.Conv2D(n_feats * 4, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.d3c2 = tf.keras.layers.Conv2D(n_feats * 4, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.d4c0 = tf.keras.layers.MaxPooling2D(2,2)
        self.d4c1 = tf.keras.layers.Conv2D(n_feats * 8, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.d4c2 = tf.keras.layers.Conv2D(n_feats * 8, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.bottleneck_c0 = tf.keras.layers.MaxPooling2D(2,2)
        self.bottleneck_c1 = tf.keras.layers.Conv2D(n_feats * 16, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.bottleneck_c2 = tf.keras.layers.Conv2D(n_feats * 16, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.u4c0 = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')
        self.u4c1 = tf.keras.layers.Conv2D(n_feats * 8, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.u4c2 = tf.keras.layers.Conv2D(n_feats * 8, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.u3c0 = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')
        self.u3c1 = tf.keras.layers.Conv2D(n_feats * 4, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.u3c2 = tf.keras.layers.Conv2D(n_feats * 4, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.u2c0 = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')
        self.u2c1 = tf.keras.layers.Conv2D(n_feats * 2, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.u2c2 = tf.keras.layers.Conv2D(n_feats * 2, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.u1c0 = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')
        self.u1c1 = tf.keras.layers.Conv2D(n_feature_map, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.u1c2 = tf.keras.layers.Conv2D(n_feature_map, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

        self.r0 = tf.keras.layers.Conv2D(n_feature_map, 3, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )
        self.r1 = tf.keras.layers.Conv2D(n_feature_map, 1, activation=tf.keras.layers.LeakyReLU(0.2), padding='same' )

    def call(self, inputs):
        """
        Args:
            inputs:     (tensor)
        Returns: 
        """
        d1c1 = self.d1c1(inputs)
        d1c2 = self.d1c2(d1c1)

        d2c0 = self.d2c0(d1c2) 
        d2c1 = self.d2c1(d2c0)
        d2c2 = self.d2c2(d2c1)

        d3c0 = self.d3c0(d2c2)  
        d3c1 = self.d3c1(d3c0)
        d3c2 = self.d3c2(d3c1)

        d4c0 = self.d4c0(d3c2)  
        d4c1 = self.d4c1(d4c0)
        d4c2 = self.d4c2(d4c1)

        bottleneck_c0 = self.bottleneck_c0(d4c2)  
        bottleneck_c1 = self.bottleneck_c1(bottleneck_c0)
        bottleneck_c2 = self.bottleneck_c2(bottleneck_c1)

        u4c0 = self.u4c0(bottleneck_c2) 
        u4c1 = self.u4c1(u4c0)
        u4c2 = self.u4c2(u4c1)

        u3c0 = self.u3c0(u4c2)  
        u3_concat = tf.concat([u3c0, d3c2], axis=-1)
        u3c1 = self.u3c1(u3_concat)
        u3c2 = self.u3c2(u3c1)

        u2c0 = self.u2c0(u3c2)  
        u2c1 = self.u2c1(u2c0)
        u2c2 = self.u2c2(u2c1)

        u1c0 = self.u1c0(u2c2)  
        u1_concat = tf.concat([u1c0, d1c2], axis=-1)
        u1c1 = self.u1c1(u1_concat)
        u1c2 = self.u1c2(u1c1)

        r0 = self.r0(u1c2)
        feature_map = self.r1(r0)

        return feature_map

