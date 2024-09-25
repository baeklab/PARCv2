import tensorflow as tf
from parc.layers.op import Advection, Diffusion
from parc.modules.smap import StyleMap
from parc.models.resnet import Resnet27, Resnet50, Resnet101, Resnet152
from parc.models.unet import Unet

class Differentiator(tf.keras.layers.Layer):
    def __init__(self, n_feats=128, resnet_blocks=0):
        super().__init__()
        if resnet_blocks==27:
            self.backbone = Resnet27()
        elif resnet_blocks==50:
            self.backbone = Resnet50()
        elif resnet_blocks==101: 
            self.backbone = Resnet101()
        elif resnet_blocks==152:
            self.backbone = Resnet152()
        elif resnet_blocks==0: 
            self.backbone = Unet()
        else: 
            raise Exception( f"resnet_blocks={resnet_blocks} not available. Please implement resnet{resnet_blocks} at  resnet.py" )

        self.T_adv = Advection()
        self.P_adv = Advection()
        self.M_adv = Advection()
        self.VX_adv = Advection()
        self.VY_adv = Advection()
        self.diffusion = Diffusion()

        self.T_map = StyleMap(n_feats)
        self.P_map = StyleMap(n_feats)
        self.M_map = StyleMap(n_feats)
        self.V_map = StyleMap(n_feats, n_out=2)

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
