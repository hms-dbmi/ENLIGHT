import h5py
import numpy as np
import json


def init_hdf5_bag(h5_path, first_patch, patch_key='imgs', coord_x_y=None, patch_extend_dim=False, coord_dim=2):
    img_patch = np.array(first_patch)
    if patch_extend_dim:
        img_patch = img_patch[np.newaxis,...]

    dtype = img_patch.dtype

    # Initialize a resizable dataset to hold the output
    img_shape = img_patch.shape
    num_patch = img_shape[0]

    maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)

    file = h5py.File(h5_path, "w")
    dset = file.create_dataset(patch_key, 
                                shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

    dset[:] = img_patch
    # dset.attrs['feat_extractor'] = 'conch'

    if coord_x_y is not None:
        coord_dset = file.create_dataset('coords', shape=(num_patch, coord_dim), maxshape=(None, coord_dim), chunks=(num_patch, coord_dim), dtype=np.int32)
        coord_dset[:] =  np.array(coord_x_y)[np.newaxis,...] if patch_extend_dim else np.array(coord_x_y)
    file.close()


def read_hdf5(h5_path, idx=None, key='imgs'):
    with h5py.File(h5_path, 'r') as f:
        # print(f['imgs'].shape)
        if idx == None:
            data = np.array(f[key])
        else:
            data = f[key][idx]
        # description = f.attrs['feat_extractor']
        # print(description)
    return data

def write_hdf5(h5_path, data, key='imgs', coords=None):
    data = np.array(data)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(key, data=data)
        if coords is not None:
            f.create_dataset('coords', data=np.array(coords, dtype=np.int32))


def read_hdf5_size(h5_path, key='imgs'):
    with h5py.File(h5_path, 'r') as f:
        size = f[key].shape
    return size


def add_hdf5_bag(h5_path, add_patch, patch_key='imgs', coord_x_y=None, patch_extend_dim=False):
    img_patch = np.array(add_patch)

    img_shape = img_patch.shape
    num_patch = img_shape[0]

    file = h5py.File(h5_path, "a")
    dset = file[patch_key]
    dset.resize(len(dset) + num_patch, axis=0)
    dset[-num_patch:,] = img_patch

    if coord_x_y is not None:
        coord_dset = file['coords']
        coord_dset.resize(len(coord_dset) + num_patch, axis=0)
        coord_dset[-num_patch:,] = np.array(coord_x_y)[np.newaxis,...] if patch_extend_dim else np.array(coord_x_y)
    file.close()


def write_dict_to_json(json_file, datadict):
    with open(json_file, 'w') as f:
        json.dump(datadict, f)

def read_dict_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
