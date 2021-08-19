#!/usr/bin/env python

import sys, numpy as np

class CSGSolid(object):
    def __init__(self, i, item):
        self.i = i 
        self.label = item[0].tobytes().decode("utf-8") 
        self.numPrim = item[1]
        self.primOffset = item[2]
        self.padding = item[3]
        self.cx = item[4].view(np.float32)
        self.cy = item[5].view(np.float32)
        self.cz = item[6].view(np.float32)
        self.ex = item[7].view(np.float32)

    def __repr__(self):
        fmt = "CSGSolid(%d) %10s numPrim:%3d primOffset:%4d center:(%10.4f,%10.4f,%10.4f) extent:%10.4f "
        return fmt % ( self.i, self.label, self.numPrim, self.primOffset, self.cx, self.cy, self.cz, self.ex )


if __name__ == '__main__':

    path = sys.argv[1]
    a = np.load(path)

    print(path)
    print(a.shape)

    for i in range(len(a)):
        so = CSGSolid(i, a[i])
        print(so)
    pass


