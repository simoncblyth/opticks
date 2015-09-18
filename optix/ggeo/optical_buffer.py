#!/usr/bin/env python

import json, os, numpy as np


if __name__ == '__main__':
    m = json.load(file(os.path.expanduser("~/.opticks/GMaterialIndexLocal.json")))
    im = dict(zip(map(int,m.values()),map(str,m.keys())))
    o = np.load(os.path.expandvars("$IDPATH/optical.npy")).reshape(-1,6,4)
    for i,b in enumerate(o):
        print "%2d %2d : %25s (%2d) %25s (%2d) " % ( i,i+1,im[b[0,0]],b[0,0], im[b[1,0]],b[1,0] )





