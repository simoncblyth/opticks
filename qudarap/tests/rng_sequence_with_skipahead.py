#!/usr/bin/env python 

import os, numpy as np, logging
log = logging.getLogger(__name__)


def rng_sequence_with_skipahead(base):
    fold = os.path.expandvars(base)
    symbols = "abcdefghijklmnopqrstuvwxyz"
    names = sorted(os.listdir(fold))

    arrs = []
    for i, name in enumerate(names):
        sym = symbols[i]
        path = os.path.join(fold, name)

        print("fold:[%s]" % fold) 
        print("name:[%s]" % name) 
        print("path:[%s]" % path) 

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
    rng_sequence_with_skipahead("$FOLD/rng_sequence_f_ni1000000_nj16_nk16_tranche100000") 
    print("seq.shape %s " % str(seq.shape))
pass


