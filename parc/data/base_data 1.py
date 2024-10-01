import numpy as np

def norm(vec):
    vsum = 0
    for vec_i in vec: 
        vsum += vec_i**2
    return np.sqrt( vsum / len(vec) ) 

class BaseData:
    def __init__(self, **kwargs):
        super(BaseData, self).__init__(**kwargs)

    # Normalization
    # todo: commeent
    def data_normalization(self, input_data, no_of_channel):
        norm_data = np.zeros(input_data.shape)
        min_val = []
        max_val = []
        for i in range(no_of_channel):
            norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - np.amin(input_data[:,:,:,i::no_of_channel])) / (np.amax(input_data[:,:,:,i::no_of_channel]) - np.amin(input_data[:,:,:,i::no_of_channel])) + 1E-9)
            min_val.append(np.amin(input_data[:,:,:,i::no_of_channel]))
            max_val.append(np.amax(input_data[:,:,:,i::no_of_channel]))
        return norm_data, min_val, max_val        
    
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

    # todo: commeent
    def data_normalization_with_value(self, input_data, min_val, max_val, no_of_channel):
        norm_data = np.zeros(input_data.shape)
        for i in range(no_of_channel):
            norm_data[:,:,:,i::no_of_channel] = ((input_data[:,:,:,i::no_of_channel] - min_val[i]) / (max_val[i] - min_val[i] + 1E-9))
        return norm_data

    # todo: commeent
    def data_denormalization(self, input_data, min_val, max_val, no_of_channel):
        denorm_data = np.zeros(input_data.shape)
        for i in range(no_of_channel):
            denorm_data[:,:,:,i::no_of_channel] = (input_data[:,:,:,i::no_of_channel] * (max_val[i] - min_val[i] + 1E-9)) + min_val[i]
        return denorm_data