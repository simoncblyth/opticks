#!/usr/bin/env python 
"""

~/o/qudarap/tests/QSimTest.sh ana
~/o/qudarap/tests/QSimTest.sh pdb


"""
import os, numpy as np, logging

np.set_printoptions(suppress=True, linewidth=200)
MODE = int(os.environ.get("MODE", "0"))
log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import os

class QSimTest(object):
    def __init__(self):
        pass

    def rng_sequence(self):
        fold = os.path.expandvars("$FOLD/rng_sequence_f_ni1000000_nj16_nk16_tranche100000")
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
    def multifilm_lut(self):
        fold = os.path.expandvars("$FOLD/multifilm_lut_result.npy")
        interp_result = np.load(fold)
        sample = np.load("$FOLD/sample.npy")
        #bnd_range = sample.shape[1]
        bnd_range = 2 
        for bnd in range(bnd_range):
            sub_interp = interp_result[interp_result[:,1] == bnd ] 
            sub_sample = sample[sample[:,1] == bnd ]
            diff = sub_interp - sub_sample
            fig, axs = plt.subplots(2,2)
            axs[0,0].scatter(sub_sample[:,4],diff[:,4],s=0.5, label="R_s")
            axs[0,1].scatter(sub_sample[:,5],diff[:,5],s=0.5, label="T_s")
            axs[1,0].scatter(sub_sample[:,6],diff[:,6],s=0.5, label="R_p")
            axs[1,1].scatter(sub_sample[:,7],diff[:,7],s=0.5, label="T_p")
            for i in range(2):
                for j in range(2):
                    axs[i,j].set_xlabel("sample value")
                    axs[i,j].set_ylabel("interp - sample")
                    axs[i,j].legend()
            plt.suptitle("boundary Type = {}".format(bnd),fontsize = 30 )
            plt.show()        
           

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QSimTest()

    TEST = os.environ.get("TEST","")
    print("TEST:%s" % TEST );   

    if TEST == "rng_sequence":
        t.rng_sequence() 
    elif TEST == "multifilm_lut":
        t.multifilm_lut()
    else:
        print("TEST UNHANDLED:%s" % TEST );   
    pass
pass
