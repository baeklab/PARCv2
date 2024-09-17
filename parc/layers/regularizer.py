import tensorflow as tf

class spade_unit(tf.keras.layers.Layer):
    def __init__(self, n_feats=128):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(n_feats, 3, padding='same', activation='relu')
        self.conv_gamma = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.conv_beta = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.epsilon = 1e-5

    def call(self, inputs):
        """
        Args: 
            inputs:     (tensor) [0] coupled feature map [1] advection and diffusion
        Returns:
            normalized: (tensor) spatially adaptive normalized by advection and diffusion
        """
        x, mask = inputs[0], inputs[1]
        
        # mask = tf.image.resize(mask, x.shape[1:3], method='nearest')
        mask = self.conv(mask)
        
        gamma = self.conv_gamma(mask)
        beta = self.conv_beta(mask)
        
        # potential bug: try BatchNorm? 
        mean, var = tf.nn.moments(x, axes=(0,1,2), keepdims=True)
        std = tf.sqrt( tf.math.add(var, self.epsilon) )
        normalized = tf.math.divide( tf.math.subtract(x, mean), std)
        
        normalized = tf.math.add( tf.math.multiply(gamma, normalized), beta )
        
        return normalized   

class spade_block(tf.keras.layers.Layer): 
    def __init__(self, n_feats=128):
        super().__init__()
        self.spade1 = spade_unit(n_feats=n_feats)
        self.spade2 = spade_unit(n_feats=n_feats)
        self.spade3 = spade_unit(n_feats=n_feats)
        
        self.conv1 = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        
    def call(self, inputs):
        """
        Args: 
            inputs:     (tensor) [0] coupled feature map [1] advectino and diffusion
        Returns:
            out:        (tensor) todo
        """
        x, mask = inputs[0], inputs[1]
        x = tf.keras.layers.GaussianNoise(0.05)(x)
        
        s1 = self.conv1( tf.keras.layers.LeakyReLU(0.2)( self.spade1( [x, mask] ) ) )
        s2 = self.conv2( tf.keras.layers.LeakyReLU(0.2)( self.spade2( [s1, mask] ) ) )    
        s_skip = self.conv3( tf.keras.layers.LeakyReLU(0.2)( self.spade3( [x, mask] ) ) )
        
        out = tf.keras.layers.add( [s2, s_skip] )
        return out    
