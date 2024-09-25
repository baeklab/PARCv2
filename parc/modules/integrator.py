import tensorflow as tf
from parc.layers.resnet import ResNetBlock
from parc.layers.regularizer import SpadeBlock

class IntegratorUnit(tf.keras.layers.Layer):
    """ data-driven integrator to correct errors of numerial integrator for higher order terms
    """ 
    def __init__(self, n_feats=128, n_out=1):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(n_feats, 1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(n_feats, 1, padding='same', activation='relu')

        self.resnet2 = ResNetBlock(filters=n_feats, n_blocks=2, stride=1, n_out=n_feats)

        self.style = SpadeBlock(n_feats=n_feats)
        self.conv = tf.keras.layers.Conv2D(n_out, 1, padding='same')

    def call(self, inputs):
        """
        Args: 
            inputs:     (tensor) todo
        Returns:
            out:        (tensor) todo
        """
        x, mask = inputs[0], inputs[1]

        x = self.conv1(x)
        x = self.conv2(x)
        style_vector = self.style(x)
        resnet_out = self.resnet2(style_vector)
        out = self.conv(resnet_out)
        return out

class Integrator(tf.keras.layers.Layer):
    def __init__(self, n_feats=128):
        super().__init__()
        self.T_int = IntegratorUnit(n_feats=n_feats)
        self.P_int = IntegratorUnit(n_feats=n_feats)
        self.M_int = IntegratorUnit(n_feats=n_feats)
        self.U_int = IntegratorUnit(n_feats=n_feats, n_out=2)
        
    def call(self, inputs):
        """
        Args: 
            inputs:     (tensor) todo
        Returns: 
            X_next      (tensor) todo 
            U_next      (tensor) todo
        """
        x_dot, u_dot, x, u = inputs[0], inputs[1], inputs[2], inputs[3]
        
        T_next = self.T_int( x[..., 0:1], x_dot[..., 0:1] )
        P_next = self.P_int( x[..., 1:2], x_dot[..., 1:2] )
        M_next = self.M_int( x[..., 2:3], x_dot[..., 2:3] )
        U_next = self.U_int( u[..., 4:6], u_dot[..., 4:6] ) # tag: double check
        X_next = keras.concat( [T_next, P_next, M_next], axis=-1 )        
        
        return tf.convert_to_tensor(X_next), tf.convert_to_tensor(U_next)
