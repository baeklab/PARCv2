import tensorflow as tf
from parc.modules.differentiator import differentiator
from parc.modules.integrator import integrator


class parcV2(tf.keras.Model):
    def __init__(self, numerical_int='fe', n_ts=1, n_step=1/40, resnet_blocks=10):
        """
        Args:
            numerical_int:  (str) numerical integrator, {'fe', 'rk4'}  
            n_ts:           (int) number of time steps to train/inference on 
            n_step:         (int) number of step size to integrate 
            resnet_blocks:  (int) backbone to extract features, {0:UNet, 27:ResNet27, 50:ResNet50, 101:ResNet101, 150:ResNet150}
        """
        super().__init__()
        self.differentiator = differentiator(resnet_blocks=resnet_blocks)
        self.integrator = integrator()
        self.numerical_int = numerical_int
        self.n_ts = n_ts
        self.n_step = n_step

    # debug with inference
    def call(self, inputs):
        # tag: copy and referenc error => make sure to copy or don't use redundant updates
        x_in, u_in = tf.cast( inputs[0], tf.float32 ), tf.cast( inputs[1], tf.float32 )
        data_in = tf.concat( [x_in, u_in], axis=-1 )

        x_preds = [None] * self.n_ts
        u_preds = [None] * self.n_ts
        for t in range(self.n_ts):
            # data_in = tf.clip_by_value(data_in, 0., 1.) # tag: suspicious

            if self.numerical_int == 'fe':
                update = self.differentiator(data_in)
            elif self.numerical_int == 'rk4':
                k1 = self.differentiator(data_in)
                k2 = tf.math.add( data_in, tf.math.multiply( tf.math.multiply( self.n_step, 0.5 ), k1 ) )
                k2 = self.differentiator(k2)

                k3 = tf.math.add( data_in, tf.math.multiply( tf.math.multiply( self.n_step, 0.5 ), k2 ) )
                k3 = self.differentiator(k3)

                k4 = tf.math.add( data_in, tf.math.multiply( self.n_step, k3 ) )
                k4 = self.differentiator(k4)

                k1 = tf.math.divide( k1, 6 )
                k2 = tf.math.divide( k2, 3 )
                k3 = tf.math.divide( k3, 3 )
                k4 = tf.math.divide( k4, 6 )

                update = tf.math.add_n( [k1, k2, k3, k4] )
            else:
                raise ValueError('please specify numerical integration')

            data_in = tf.math.add( data_in, tf.math.multiply( self.n_step, update ) )
            x_preds[t] = data_in[..., :3]
            u_preds[t] = data_in[..., 3:]

        return tf.concat( x_preds, axis=-1 ), tf.concat( u_preds, axis=-1 )


    def train_step(self, data):
        data_in, y_true = data
        X_true, U_true = y_true

        X_true = tf.cast( X_true, tf.float32 )
        U_true = tf.cast( U_true, tf.float32 )

        with tf.GradientTape() as tape:
            X_pred, U_pred = self(data_in)
            loss = tf.math.reduce_sum( tf.math.abs( tf.math.subtract( X_true, X_pred  ) ) ) + tf.math.reduce_sum( tf.math.abs( tf.math.subtract( U_true, U_pred ) ) )

        trainable_weights = self.trainable_variables
        grads = tape.gradient( loss, trainable_weights )
        self.optimizer.apply_gradients( zip(grads, trainable_weights) )

        return {'loss' : loss}
