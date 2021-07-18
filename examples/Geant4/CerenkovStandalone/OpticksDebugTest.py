#!/usr/bin/env python

import os, numpy as np

def np_concatenate(fold):
    aa = []
    for name in os.listdir(fold):
        if not name.endswith(".npy"): continue 
        path = os.path.join(fold, name)
        a = np.load(path)
        aa.append(a)
    pass
    c = np.concatenate(aa)
    return c 


if __name__ == '__main__':
    a_dir="/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
    b_path = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche1000000/rng_sequence_f_ni1000000_nj16_nk16_ioffset000000.npy"

    a = np_concatenate(a_dir)
    b = np.load(b_path)
    assert np.all( a == b )
    


