#!/usr/bin/env python
"""

ipython -i tests/X4PhysicalConstantsTest.py

"""
import os, numpy as np
np.set_printoptions(suppress=False)  

if __name__ == '__main__':
    FOLD = os.path.expandvars("$TMP/X4PhysicalConstantsTest")
    names = os.listdir(FOLD)
    tags = "abcdefghijklmnopqrstuvwxyz" 
    for i, name in enumerate(filter(lambda n:n[-4:] == ".npy", names)):


        tag = tags[i]
        path = os.path.join(FOLD, name)
        txtpath = os.path.join(FOLD, name.replace(".npy",".txt"))
        arr = np.load(path)

        if os.path.exists(txtpath):
            txt = open(txtpath, "r").readlines() 
        else:
            txt = []
        pass 
        print("tag %s path %s arr %s " % (tag, path, str(arr.shape)))
        print(txt)

        globals()[tag] = arr
    pass

