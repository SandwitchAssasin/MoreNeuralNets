import struct
import numpy as np
def turn_int_into_one_hot(X):
    #accepts only vectors as input
    m = np.max(X) + 1
    S = np.zeros(shape=(len(X),m))
    for i in range(len(X)):
        S[i,X[i]] = 1
    return S


def read_idx(filename):
    with open(filename,'rb') as f:
        magic, = struct.unpack('>I',f.read(4))
        data_type = (magic>>8)&0xFF
        num_dims = magic&0xFF

        shape = tuple(struct.unpack('>I',f.read(4))[0] for _ in range(num_dims))

        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64,
            }
        if data_type not in dtype_map:
            raise Exception('Unsupported data type')
        data = np.frombuffer(f.read(),dtype=dtype_map[data_type])
        return data.reshape(shape)
