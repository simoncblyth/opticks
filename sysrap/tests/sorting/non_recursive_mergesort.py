#!/usr/bin/env python

import numpy as np


if __name__ == '__main__':

    dir_ = "/tmp"
    apath = os.path.join(dir_, "a.npy")
    bpath = os.path.join(dir_, "b.npy")
  
    if not os.path.exists(apath):
        a = np.random.sample(1024).astype(np.float32)    
        print("Creating random sample a %s saved to %s " % (str(a.shape), apath))
        np.save(apath, a )
    else:
        a = np.load(apath)
        b = np.load(bpath)
        print("Loaded a %s from %s " % (str(a.shape), apath))
        print("Loaded b %s from %s " % (str(b.shape), bpath))
    pass
pass






