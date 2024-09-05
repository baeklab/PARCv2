import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import os.path as osp
from datetime import datetime
import pickle

from PARC.data.data_em import *
import parc_refactor


def main(model_name, dir_save, numerical_int='fe', resnet_blocks=0, n_ts=1, epochs=500, batch_size=8, diff=None):
    """
    Inputs: 
        model_name:      (str)   name of model to save
        dir_save:        (dirs)  directory to save model
        numerical_int:   (str)   numerical integrator, {'fe', 'rk4'} for Forward Euler and RK4
        resnet_blocks:   (int)   backbone feature extractor, {0:Unet, 27:ResNet27, 50:ResNet50, 127:ResNet127} # todo: make it more dynamic
        n_ts:            (int)   number of time-steps to make a prediction
        epochs:          (int)   number of epochs to train
        batch_size:      (int)   number of batch size
        diff:            (dir)   directory to pre-trained differentiator
    """
    curr_datetime = datetime.now().strftime( "%m%d_%H%M" )
    print( f"Running 'parc_refactor.py at {curr_datetime} for {epochs} saving as {model_name} at {dir_save}", flush=True )
    print( f"\tPARC with resnet_blocks of {resnet_blocks} with {numerical_int} integrator for {n_ts} temporal dynamcis", flush=True )
    if not osp.exists(dir_save):
        print( f"{dir_save} does not exist", flush=True)
    
    """ this should be per-user
     x: (cases, X, Y, number of fields), fields
     u: (cases, X, Y, 2), velocity
    """
    # Get data and normalization
    data = DataEnergeticMaterials()
    x, u, _ = data.clip_raw_data(dataset_range = (0,100), dir_dataset="/project/vil_baek/data/physics/todo_single_void_data", dim_reduce=8, n_seq=n_ts+1)
    x_norm, u_norm = data.data_normalization(x,3), data.data_normalization(u,2)
    x_in = tf.data.Dataset.from_tensor_slices( (x_norm[0][...,:3], u_norm[0][...,:2]) )
    y_in = tf.data.Dataset.from_tensor_slices( (x_norm[0][...,3:], u_norm[0][...,2:]) )
    ds = tf.data.Dataset.zip((x_in, y_in))
    ds = ds.shuffle(buffer_size = 2192) 
    ds = ds.batch(batch_size)

    # prepare data
    tf.keras.backend.clear_session()
    parc = parc_refactor_resnet.parcV2(n_ts=n_ts, numerical_int=numerical_int, resnet_blocks=resnet_blocks)

    if diff is not None: 
        parc.predict( [x_norm[0][..., :3], u_norm[0][..., :2]] )
        parc.load_weights(diff)
    
    parc.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999) )
    history = parc.fit(ds, epochs = epochs, shuffle = True)

    parc.save_weights( osp.join( dir_save, model_name + '_' + curr_datetime + '.h5' ) )

    with open( osp.join(dir_save, model_name + '_' + curr_datetime + '.pkl'), 'wb' ) as f: 
        pickle.dump(history.history, f)
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='parc refactored')
    parser.add_argument('--name') 
    parser.add_argument('--dir_save')
    parser.add_argument('--integrator', choices=['fe', 'rk4'])
    parser.add_argument('--resnet_blocks', type=int)
    parser.add_argument('--n_ts', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--diff', default=None)
    parser.add_argument('--version', default=None)
    args = parser.parse_args()
    main(model_name=args.name, dir_save=args.dir_save, numerical_int=args.integrator, resnet_blocks=args.resnet_blocks, n_ts=args.n_ts, epochs=args.epochs, batch_size=args.batch_size, diff=args.diff)
