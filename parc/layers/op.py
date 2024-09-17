import tensorflow as tf

class advection(tf.keras.layers.Layer):
    def __init__(self): 
        super().__init__()
        
    def call(self, inputs):
        """ 
        Args: 
            x:          (tensor) fields
            u:          (tensor) velocity
        Return: 
            advection:  (tensor) 
        """
        x, u = inputs[0], inputs[1]
        dy, dx = tf.image.image_gradients(x)
        # spatial_deriv = tf.concat( [dy, dx], axis=-1 )
        spatial_deriv = tf.concat( [dx, dy], axis=-1 )
        advect = tf.reduce_sum( tf.multiply( spatial_deriv, u ), axis=-1, keepdims=True )
        return advect

class diffusion(tf.keras.layers.Layer):
    def __init__(self): 
        super().__init__()
        
    def call(self, inputs):
        """ 
        Args: 
            inputs:     (tensor) fields
        Return:
            diffusion:  (tensor)
        """
        dy, dx = tf.image.image_gradients(inputs)
        dyy, _ = tf.image.image_gradients(dy)
        _, dxx = tf.image.image_gradients(dx)
        laplacian = tf.math.add( dyy, dxx )
        return laplacian
