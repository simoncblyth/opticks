#!/usr/bin/env python

import sys, numpy as np

class CSGSolid(object):
    def __init__(self, i, item):
        self.i = i 
        self.label = item[:4].tobytes().decode("utf-8")   
        ## smth funny about label, needs truncation at \0 perhaps ?

        self.numPrim = item[4]
        self.primOffset = item[5]
        self.type = item[6]
        self.padding = item[7]

        self.cx = item[8].view(np.float32)
        self.cy = item[9].view(np.float32)
        self.cz = item[10].view(np.float32)
        self.ex = item[11].view(np.float32)

    def __repr__(self):
        fmt = "CSGSolid(%d) numPrim:%3d primOffset:%4d center:(%10.4f,%10.4f,%10.4f) extent:%10.4f  label [%-20s] "
        return fmt % ( self.i, self.numPrim, self.primOffset, self.cx, self.cy, self.cz, self.ex, self.label )


if __name__ == '__main__':

    path = sys.argv[1]
    a = np.load(path)

    print(path)
    print(a.shape)

    for i in range(len(a)):
        so = CSGSolid(i, a[i])
        print(so)
    pass


