import tensorflow as tf

class spade_unit(tf.keras.layers.Layer):
    def __init__(self, n_feats=128):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(n_feats, 3, padding='same', activation='relu')
        self.conv_gamma = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.conv_beta = tf.keras.layers.Conv2D(n_feats, 3, padding='same')
        self.epsilon = 1e-5

    def call(self, inputs):
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

class integrator_unit(tf.keras.layers.Layer):
    def __init__(self, n_feats=128, n_out=1):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(n_feats, 1, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(n_feats, 1, padding='same', activation='relu')
        
        self.resnet2 = resnet_block(filters=n_feats, n_blocks=2, stride=1, n_out=n_feats)
        
        self.style = spade_block(n_feats=n_feats)
        self.conv = tf.keras.layers.Conv2D(n_out, 1, padding='same')
    
    def call(self, inputs):
        x, mask = inputs[0], inputs[1]
        
        x = self.conv1(x)
        x = self.conv2(x)
        style_vector = self.style(x)
        resnet_out = self.resnet2(style_vector)
        out = self.conv(resnet_out)
        return out

# phone calls it a "style vector", ref: spade_generator_unit
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
        x, mask = inputs[0], inputs[1]
        x = tf.keras.layers.GaussianNoise(0.05)(x)
        
        s1 = self.conv1( tf.keras.layers.LeakyReLU(0.2)( self.spade1( [x, mask] ) ) )
        s2 = self.conv2( tf.keras.layers.LeakyReLU(0.2)( self.spade2( [s1, mask] ) ) )    
        s_skip = self.conv3( tf.keras.layers.LeakyReLU(0.2)( self.spade3( [x, mask] ) ) )
        
        out = tf.keras.layers.add( [s2, s_skip] )
        return out    

class style_map(tf.keras.layers.Layer):
    def __init__(self, n_feats=128, n_out=1): 
        super().__init__()
        self.style = spade_block(n_feats)
        self.resnet = resnet_block(filters=n_feats, n_blocks=2, stride=1, n_out=n_feats)
        self.conv = tf.keras.layers.Conv2D(n_out, n_out, padding='same') # bug: output dimension     
        
    def call(self, inputs):
        x, mask = inputs[0], inputs[1]
        
        style_vector = self.style( [x, mask] )
        resnet_out = self.resnet(style_vector)
        conv_out = self.conv(resnet_out)
        return conv_out    
    
class advection(tf.keras.layers.Layer):
    def __init__(self): 
        super().__init__()
        
    def call(self, inputs):
        x, u = inputs[0], inputs[1]
        dy, dx = tf.image.image_gradients(x)
        spatial_deriv = tf.concat( [dy, dx], axis=-1 )
        advect = tf.reduce_sum( tf.multiply( spatial_deriv, u ), axis=-1, keepdims=True )
        return advect

class diffusion(tf.keras.layers.Layer):
    def __init__(self): 
        super().__init__()
        
    def call(self, inputs):
        dy, dx = tf.image.image_gradients(inputs)
        dyy, _ = tf.image.image_gradients(dy)
        _, dxx = tf.image.image_gradients(dx)
        laplacian = tf.math.add( dyy, dxx )
        return laplacian

class differentiator(tf.keras.layers.Layer):
    def __init__(self, n_feats=128, resnet_blocks=0): 
        super().__init__()
        if resnet_blocks==50:
            self.backbone = resnet50()
        elif resnet_blocks==27:
            self.backbone = resnet27()
        
        self.T_adv = advection()
        self.P_adv = advection()
        self.M_adv = advection()
        self.VX_adv = advection()
        self.VY_adv = advection()
        self.diffusion = diffusion()

        self.T_map = style_map(n_feats)
        self.P_map = style_map(n_feats)
        self.M_map = style_map(n_feats) 
        self.V_map = style_map(n_feats, n_out=2)
        
    def call(self, inputs):
        T_in = inputs[..., 0:1]
        P_in = inputs[..., 1:2]
        M_in = inputs[..., 2:3]
        Vx_in = inputs[..., 3:4]
        Vy_in = inputs[..., 4:5]
        U_in = tf.concat( [Vx_in, Vy_in], axis=-1 )
        
        feature_map = self.backbone(inputs)
        
        T_adv = self.T_adv( [T_in, U_in] )
        T_diff = self.diffusion( T_in )
        T = tf.concat( [T_adv, T_diff], axis=-1 )
        T_dot = self.T_map( [feature_map, T] )
        
        P_adv = self.P_adv( [P_in, U_in] )
        P_dot = self.P_map( [feature_map, P_adv] )
        
        M_adv = self.M_adv( [M_in, U_in] )
        M_dot = self.M_map( [feature_map, M_adv] )
        
        VX_adv = self.VX_adv( [Vx_in, U_in] )
        VY_adv = self.VY_adv( [Vy_in, U_in] )
        V_adv = tf.concat( [VX_adv, VY_adv], axis=-1 )
        V_dot = self.V_map( [feature_map, V_adv] )
        
        out = tf.concat( [T_dot, P_dot, M_dot, V_dot], axis=-1 )
        return out
    
class integrator(tf.keras.layers.Layer):
    def __init__(self, n_feats=128):
        super().__init__()
        self.T_int = integrator_unit(n_feats=n_feats)
        self.P_int = integrator_unit(n_feats=n_feats)
        self.M_int = integrator_unit(n_feats=n_feats)
        self.V_int = integrator_unit(n_feats=n_feats, n_out=2)
        
    def call(self, inputs):
        x_dot, u_dot, x, u = inputs[0], inputs[1], inputs[2], inputs[3]
        
        T_next = self.T_int( x[..., 0:1], x_dot[..., 0:1] )
        P_next = self.P_int( x[..., 1:2], x_dot[..., 1:2] )
        M_next = self.M_int( x[..., 2:3], x_dot[..., 2:3] )
        V_next = self.V_int( u[..., 4:6], u_dot[..., 4:6] ) # tag: double check
        X_next = keras.concat( [T_next, P_next, M_next], axis=-1 )        
        
        return tf.convert_to_tensor(X_next), tf.convert_to_tensor(V_next)
    
class unet(tf.keras.layers.Layer):
    def __init__(self, n_feats=64, n_feature_map=128): 
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
        d1c1 = self.d1c1(inputs)
        d1c2 = self.d1c2(d1c1)
        
        d2c0 = self.d2c0(d1c2) # down
        d2c1 = self.d2c1(d2c0)
        d2c2 = self.d2c2(d2c1)
        
        d3c0 = self.d3c0(d2c2) # down
        d3c1 = self.d3c1(d3c0)
        d3c2 = self.d3c2(d3c1)

        d4c0 = self.d4c0(d3c2) # down
        d4c1 = self.d4c1(d4c0)
        d4c2 = self.d4c2(d4c1)
        
        bottleneck_c0 = self.bottleneck_c0(d4c2) # down
        bottleneck_c1 = self.bottleneck_c1(bottleneck_c0)
        bottleneck_c2 = self.bottleneck_c2(bottleneck_c1)
        
        u4c0 = self.u4c0(bottleneck_c2) #up
        u4c1 = self.u4c1(u4c0)
        u4c2 = self.u4c2(u4c1)
        
        u3c0 = self.u3c0(u4c2) # up
        u3_concat = tf.concat([u3c0, d3c2], axis=-1)
        u3c1 = self.u3c1(u3_concat)
        u3c2 = self.u3c2(u3c1)
        
        u2c0 = self.u2c0(u3c2) # up
        u2c1 = self.u2c1(u2c0)
        u2c2 = self.u2c2(u2c1)
        
        u1c0 = self.u1c0(u2c2) # up
        u1_concat = tf.concat([u1c0, d1c2], axis=-1)
        u1c1 = self.u1c1(u1_concat)
        u1c2 = self.u1c2(u1c1)
        
        r0 = self.r0(u1c2)
        feature_map = self.r1(r0)
        
        return feature_map

class parcV2(tf.keras.Model):
    def __init__(self, numerical_int='fe', n_ts=36, n_step=1/40, resnet_blocks=10):
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
            data_in = tf.clip_by_value(data_in, 0., 1.) # tag: suspicious
    
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
    
class resnet_unit(tf.keras.layers.Layer):
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
    
class resnet_block(tf.keras.layers.Layer):
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


class resnet27(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = resnet_block(filters=64, n_blocks=2, stride=1)
        self.block2 = resnet_block(filters=128, n_blocks=2, stride=1)
        self.block3 = resnet_block(filters=256, n_blocks=3, stride=1)
        self.block4 = resnet_block(filters=512, n_blocks=2, stride=1, n_out=128) # bug: make it dynamic
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out
    
class resnet50(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = resnet_block(filters=64, n_blocks=3, stride=1)
        self.block2 = resnet_block(filters=128, n_blocks=4, stride=1)
        self.block3 = resnet_block(filters=256, n_blocks=6, stride=1)
        self.block4 = resnet_block(filters=512, n_blocks=3, stride=1, n_out=128) # bug: make it dynamic
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out

class resnet101(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = resnet_block(filters=64, n_blocks=3, stride=1)
        self.block2 = resnet_block(filters=128, n_blocks=4, stride=1)
        self.block3 = resnet_block(filters=256, n_blocks=23, stride=1)
        self.block4 = resnet_block(filters=512, n_blocks=3, stride=1, n_out=128)
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out

class resnet152(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.block1 = resnet_block(filters=64, n_blocks=3, stride=1)
        self.block2 = resnet_block(filters=128, n_blocks=8, stride=1)
        self.block3 = resnet_block(filters=256, n_blocks=36, stride=1)
        self.block4 = resnet_block(filters=512, n_blocks=3, stride=1, n_out=128)
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = self.block4(x3)
        return out
            