#!/usr/bin/env python 
"""

ipython -i tests/QSimTest.py 


"""
import numpy as np, logging
log = logging.getLogger(__name__)

class QSimTest(object):
    def __init__(self):
        pass

    def rng_sequence(self):
        fold = os.path.expandvars("/tmp/$USER/opticks/QSimTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000")
        symbols = "abcdefghijklmnopqrstuvwxyz"
        names = sorted(os.listdir(fold))

        arrs = []
        for i, name in enumerate(names):
            sym = symbols[i]
            path = os.path.join(fold, name)
            arr = np.load(path)
            arrs.append(arr)
            msg = "%3s %20s %s" % (sym, str(arr.shape), path) 
            logging.info(msg)
            globals()[sym] = arr 
        pass
        seq = np.concatenate(arrs) 
        globals()["seq"] = seq
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QSimTest()
    t.rng_sequence() 


