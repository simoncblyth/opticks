#!/usr/bin/env python
"""
CSGFoundryLoadTest.py
=======================

This reviews python access to CSGSolid/CSGPrim/CSGNode 
to assist with development of importing those from stree/snd:: 

    cf.prim.view(np.int32)[:,:2].reshape(-1,8)  

    In [2]: expr
    Out[2]: 'cf.prim.view(np.int32)[:,:2].reshape(-1,8) '

    In [3]: eval(expr)
    Out[3]: 
    array([[    1,     0,     0,     0,     0,   149,     0,     0],
           [    1,     1,     1,     0,     1,    17,     0,     1],
           [    1,     2,     2,     0,     2,     2,     0,     2],
           [    3,     3,     3,     0,     3,     1,     0,     3],
           [    3,     6,     5,     0,     4,     0,     0,     4],

                nn    no      to    po     so     mx     rx     px

    nn:numNode
    no:nodeOffset
    to:tranOffset
    po:planOffset

    so:sbtIndexOffset
    mx:meshIdx  : corresponds to lvIdx
    rx:repeatIdx
    px:primIdx  

    nn = cf.prim.view(np.int32)[:,0,0]

    In [7]: np.unique( nn, return_counts=True )
    Out[7]: 
    (array([  1,   3,   7,  15,  31, 127], dtype=int32),
     array([ 933,  127, 2130,   12,    1,   56]))

    ## The nn:numNode are all complete binary tree sizes 
    ## Presumably that means there are no multiunions with subNum/subOffset in use ? 

    In [9]: cf.prim.view(np.int32)[:,0,0].sum()
    Out[9]: 23547

    In [10]: cf.node.shape
    Out[10]: (23547, 4, 4)

    In [66]: mx = cf.prim.view(np.int32)[:,1,1] 
    In [71]: mx.min(), mx.max()
    Out[71]: (0, 149)

    In [63]: rx = cf.prim.view(np.int32)[:,1,2]  # repeatIndex associating the prim to its solid 
    In [65]: np.unique(rx, return_counts=True )
    Out[65]: 
    (array([0,       1,    2,    3,    4,    5,    6,    7,    8,    9], dtype=int32),
     array([3089,    5,   11,   14,    6,    1,    1,    1,    1,  130]))

    ## Those unique ridx counts correspond directly to the solid numPrim

    In [98]: cf.solid[:,1,0]
    Out[98]: array([3089,    5,   11,   14,    6,    1,    1,    1,    1,  130], dtype=int32)

    In [72]: px = cf.prim.view(np.int32)[:,1,3]
    In [76]: px.min(), px.max(), len(px)
    Out[76]: (0, 3088, 3259)


    In [87]: np.all( np.arange(3089) == px[:3089] )   
    Out[87]: True
    # contiguous monotonic up to 3089 : Those are the primIdx for the ridx:0 globals

Beyond 3089 the last 170 prim dont increment the primIdx. 
Thats because they are the instances, that reset the primIdx back to zero for each::

    In [94]: px[3089:]
    Out[94]: 
    array([
             0,   1,   2,   3,   4,                                                          # ridx:1  (5)
             0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,                            # ridx:2  (11)  
             0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,             # ridx:3  (14)
             0,   1,   2,   3,   4,   5,                                                     # ridx:4  (6)  
             0,                                                                              # ridx:5  (1)
             0,                                                                              # ridx:6  (1)
             0,                                                                              # ridx:7  (1)
             0,                                                                              # ridx:8  (1)
             0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,   # ridx:9  (130)
            16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  
            32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  
            48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  
            64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  
            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  
            96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
           112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 
           128, 129]
           , dtype=int32)

    In [95]: px[3089:].shape
    Out[95]: (170,)

Looking at cf.solid makes this plain::

    In [51]: cf.solid.view(np.int32)[:,0].copy().view("|S16")
    Out[51]: 
    array([[b'r0'],
           [b'r1'],
           [b'r2'],
           [b'r3'],
           [b'r4'],
           [b'r5'],
           [b'r6'],
           [b'r7'],
           [b'r8'],
           [b'r9']], dtype='|S16')

    In [56]: cf.solid.view(np.int32)[:,1]
    Out[56]: 
    array([[3089,    0,    0,    0],
           [   5, 3089,    0,    0],
           [  11, 3094,    0,    0],
           [  14, 3105,    0,    0],
           [   6, 3119,    0,    0],
           [   1, 3125,    0,    0],
           [   1, 3126,    0,    0],
           [   1, 3127,    0,    0],
           [   1, 3128,    0,    0],
           [ 130, 3129,    0,    0]], dtype=int32)

         numPrim primOffset 

    In [58]: cf.solid.view(np.int32)[:,1,0].sum()
    Out[58]: 3259

    In [59]: cf.prim.shape
    Out[59]: (3259, 4, 4)


Associating each prim to its nodes::

    In [102]: pnn = cf.prim.view(np.int32)[:,0,0]
    In [103]: pno = cf.prim.view(np.int32)[:,0,1]



* HMM associating the nodes to each solid 

Start of the node index is contiguous monotonic::

    In [14]: ix = cf.node.view(np.int32)[:,1,3] 
    In [36]: np.all( np.arange(23200) == ix[:23200] )
    Out[36]: True
    In [37]: ix.shape
    Out[37]: (23547,)
    In [40]: np.all( np.arange(23207) == ix[:23207] )
    Out[40]: True
    In [41]: np.all( np.arange(23208) == ix[:23208] )
    Out[41]: False

"""

import numpy as np, logging
log = logging.getLogger(__name__)
from opticks.CSG.CSGFoundry import CSGFoundry 

np.set_printoptions(edgeitems=16)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cf = CSGFoundry.Load()
    print(repr(cf))

    expr = "cf.prim.view(np.int32)[:,:2].reshape(-1,8) "
    print(expr)
    eval(expr)

    #print(cf.descSolid(1))
    print(cf.descSolids(True))
    print(cf.descSolids(False))

    


