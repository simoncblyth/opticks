#!/usr/bin/env python 
import numpy as np, logging
log = logging.getLogger(__name__)


def rng_sequence(base):
    fold = os.path.expandvars(base)
    symbols = "abcdefghijklmnopqrstuvwxyz"
    names = sorted(os.listdir(fold))

    arrs = []
    for i, name in enumerate(names):
        sym = symbols[i]
        path = os.path.join(fold, name)
        arr = np.load(path)
        arrs.append(arr)
        msg = "%3s %20s %s" % (sym, str(arr.shape), path) 
        print(msg)
        globals()[sym] = arr 
    pass
    seq = np.concatenate(arrs) 
    globals()["seq"] = seq
pass


if __name__ == '__main__':
    base = "/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
    rng_sequence(base) 
    print("seq.shape %s " % str(seq.shape))
pass


