#!/usr/bin/env python
"""
stree_load_test.py
====================

::

    In [14]: np.unique( st.rem.num_child, return_index=True, return_counts=True  )
    Out[14]: 
    (array([    0,     1,     2,     3,     4,    54,    63,  4521, 46276], dtype=int32),
     array([   4,    2,    0,    6,   14, 2328,   12,  204, 2326]),
     array([2884,    7,   65,    3,  126,    1,    1,    1,    1]))


    In [18]: np.unique( st.rem.lvid, return_counts=True  )
    Out[18]: 
    (array([  0,   1,   2,   3,   4,   5,   6,   7,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
             42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
             80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97, 102, 103, 123, 124, 125, 126, 127, 128, 135, 136, 137, 138], dtype=int32),
     array([  1,   1,   1,   1,   1,   1,   1,   1, 126,  63,   1,   1,   1,   1,  10,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  10,  30,  30,
             30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,
             30,  30,  30,  30,  30,  30,  30,  30,  30,  30,   2,  36,   8,   8,   1,   1, 370, 220,  56,  56,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]))



    

    In [8]: np.unique(f.nds[:,11], return_counts=True )  # repeat_index:11
    Out[8]: 
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32),
     array([  3089, 128000,  88305,  34979,  14400,    590,    590,    590,    590,  65520]))


    In [9]: snode.Label(6,11), f.nds[f.nds[:,ri] == 1 ]
    Out[9]: 
    ('           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      bd',
     array([[194249,      6,  20675,  67846,      2, 194250, 194254,    122, 300000,     -1,     -1,      1,     26],
            [194250,      7,      0, 194249,      2, 194251, 194253,    120,      0,     -1,     -1,      1,     29],
            [194251,      8,      0, 194250,      0,     -1, 194252,    118,      0,     -1,     -1,      1,     36],
            [194252,      8,      1, 194250,      0,     -1,     -1,    119,      0,     -1,     -1,      1,     37],
            [194253,      7,      1, 194249,      0,     -1,     -1,    121,      0,     -1,     -1,      1,     24],
            [194254,      6,  20676,  67846,      2, 194255, 194259,    122, 300001,     -1,     -1,      1,     26],
            [194255,      7,      0, 194254,      2, 194256, 194258,    120,      0,     -1,     -1,      1,     29],
            [194256,      8,      0, 194255,      0,     -1, 194257,    118,      0,     -1,     -1,      1,     36],
            [194257,      8,      1, 194255,      0,     -1,     -1,    119,      0,     -1,     -1,      1,     37],
            [194258,      7,      1, 194254,      0,     -1,     -1,    121,      0,     -1,     -1,      1,     24],



    n = st.f.csg.node

    In [8]: snd.Label(3,8), n[n[:,snd.lv] == 105]
    Out[8]: 
    ('        ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf',
     array([[483,   4,   0, 484,   0,  -1,  -1, 105, 105, 294, 294,  -1],
            [484,   3,   0, 486,   1, 483, 485, 105,  11,  -1,  -1,  -1],
            [485,   3,   1, 486,   0,  -1,  -1, 105, 103, 295, 295, 183],
            [486,   2,   0, 488,   2, 484, 487, 105,   1,  -1,  -1,  -1],
            [487,   2,   1, 488,   0,  -1,  -1, 105, 105, 296, 296, 184],
            [488,   1,   0, 495,   2, 486, 494, 105,   1,  -1,  -1,  -1],
            [489,   4,   0, 490,   0,  -1,  -1, 105, 105, 297, 297,  -1],
            [490,   3,   0, 492,   1, 489, 491, 105,  11,  -1,  -1,  -1],
            [491,   3,   1, 492,   0,  -1,  -1, 105, 103, 298, 298, 186],
            [492,   2,   0, 494,   2, 490, 493, 105,   1,  -1,  -1,  -1],
            [493,   2,   1, 494,   0,  -1,  -1, 105, 105, 299, 299, 187],
            [494,   1,   1, 495,   2, 492,  -1, 105,   1,  -1,  -1, 188],
            [495,   0,  -1,  -1,   2, 488,  -1, 105,   3,  -1,  -1,  -1]], dtype=int32))


    In [9]: print(st.desc_csg(18))
    desc_csg lvid:18 st.f.soname[18]:GLw1.up10_up11_FlangeI_Web_FlangeII0x59f4850 
            ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf
    array([[ 32,   2,   0,  34,   0,  -1,  33,  18, 110,  25,  25,  -1],
           [ 33,   2,   1,  34,   0,  -1,  -1,  18, 110,  26,  26,   5],
           [ 34,   1,   0,  36,   2,  32,  35,  18,   1,  -1,  -1,  -1],
           [ 35,   1,   1,  36,   0,  -1,  -1,  18, 110,  27,  27,   6],
           [ 36,   0,  -1,  -1,   2,  34,  -1,  18,   1,  -1,  -1,  -1]], dtype=int32)



"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree, snode, snd

np.set_printoptions(edgeitems=16)

def test_bd(st):
    print("u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True ) ")
    u_bd, n_bd = np.unique( st.nds.boundary, return_counts=True ) 
    for i in range(len(u_bd)):
        u = u_bd[i]
        n = n_bd[i]
        print(" %3d : %4d : %6d : %s " % (i, u, n, st.f.bd_names[u] )) 
    pass
    print(st.desc_boundary())

def test_rem(st):
    print("u_lv, n_lv = np.unique( st.rem.lvid, return_counts=True  )") 
    u_lv, n_lv = np.unique( st.rem.lvid, return_counts=True  )
    for i in range(len(u_lv)):
        u = u_lv[i]
        n = n_lv[i]
        print(" %3d : %4d : %6d : %s " % (i, u, n, st.f.soname[u] )) 
    pass

    print(st.desc_remainder())


def test_csg(st):
    print("[---test_csg" ) 
    print(" count total csg nodes for each lv : ie number of nodes in every subtree ")
    print("u_cl, n_cl = np.unique( st.f.csg.node[:,snd.lv], return_counts=True)")
    u_cl, n_cl = np.unique( st.f.csg.node[:,snd.lv], return_counts=True)   
    for i in range(len(u_cl)):
        u = u_cl[i]     
        n = n_cl[i]     
        print(" %3d : %4d : %6d : %s " % (i, u, n, st.f.soname[u] )) 
    pass
    print("]---test_csg" ) 


if __name__ == '__main__':

    snode.Type()
    snd.Type()
    

    print("[--f = Fold.Load")
    f = Fold.Load(symbol="f")
    print("]--f = Fold.Load")

    print("[--repr(f)")
    print(repr(f))
    print("]--repr(f)")

    print("[--st = stree(f)")
    st = stree(f)
    print("]--st = stree(f)")

    print("[--repr(st)")
    print(repr(st))
    print("]--repr(st)")

    #test_bd(st)
    #test_rem(st)

    test_csg(st)



