import numpy as np

def norm(vec):
    vsum = 0
    for vec_i in vec: 
        vsum += vec_i**2
    return np.sqrt( vsum / len(vec) ) 

class BaseData:
    def __init__(self, **kwargs):
        super(BaseData, self).__init__(**kwargs)

    def data_normalization(self, x, n_channel):
        norm_data = np.zeros(x.shape)
        vmins = []
        vmaxs = [] #fixed size
        for i in range(n_channel):
            norm_data[:,:,:,i::n_channel] = ((x[:,:,:,i::n_channel] - np.amin(x[:,:,:,i::n_channel])) / (np.amax(x[:,:,:,i::n_channel]) - np.amin(x[:,:,:,i::n_channel])) + 1E-9)
            vmins.append(np.amin(x[:,:,:,i::n_channel]))
            vmaxs.append(np.amax(x[:,:,:,i::n_channel]))
        return norm_data, vmins, vmaxs        
    
    def vector_normalization(self, u):
        vx = u[..., 0]
        vy = u[..., 1]
        vx_next = u[..., 2]
        vy_next = u[..., 3]
        
        vs = [ [x,y] for x,y in zip( np.ndarray.flatten(vx), np.ndarray.flatten(vy) ) ] # construct vector
        norms = [ norm(x) for x in vs ]
        max_idx = np.argmax( norms ) 
        max_vec = vs[max_idx]
        print( f"Max Vec: {max_vec}", flush=True )

        vx_flat = np.array( [x / max_vec[0] for x in np.ndarray.flatten(vx)] )
        vy_flat = np.array( [x / max_vec[1] for x in np.ndarray.flatten(vy)] ) 
        vx_next_flat = np.array( [x / max_vec[0] for x in np.ndarray.flatten(vx_next)] )
        vy_next_flat = np.array( [x / max_vec[1] for x in np.ndarray.flatten(vy_next)] )

        vx_flat.resize( vx.shape )
        vy_flat.resize( vy.shape )
        vx_next_flat.resize( vx.shape ) 
        vy_next_flat.resize( vy.shape )
        return np.stack( (vx_flat, vy_flat, vx_next_flat, vy_next_flat), axis=-1)
