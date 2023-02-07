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

    In [13]: np.unique( st.nds.repeat_index, return_counts=True )
    Out[13]: 
    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32),
     array([  3089, 128000, 138765,  69958,  14400,    590,    590,    590,    590,  65520]))


    In [17]: np.unique( st.nds.lvid[st.nds.repeat_index == 1], return_index=True, return_counts=True )
    Out[17]: 
    (array([129, 130, 131, 132, 133], dtype=int32),
     array([2, 3, 1, 4, 0]),                        ## index starts from zero, because have selected only the one ridx 
     array([25600, 25600, 25600, 25600, 25600]))

    In [22]: st.nds.lvid[st.nds.repeat_index == 1].reshape(-1,5)
    Out[22]: 
    array([[133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],
           [133, 131, 129, 130, 132],


    In [8]: np.unique( st.csg.typecode, return_counts=True )
    Out[8]: 
    (array([  1,   2,   3,  11, 101, 103, 105, 108, 110], dtype=int32),
     array([199,   1,  30,  20,   7,  27,  97,  10, 246]))






    CSGFoundry.dumpSolid ridx  1 label               r1 numPrim      5 primOffset   3089 
     so_primIdx 3089 numNode    3 nodeOffset 23207 meshIdx 133 repeatIdx   1 primIdx2   0 : PMT_3inch_pmt_solid 
     so_primIdx 3090 numNode    1 nodeOffset 23210 meshIdx 131 repeatIdx   1 primIdx2   1 : PMT_3inch_body_solid_ell_ell_helper 
     so_primIdx 3091 numNode    1 nodeOffset 23211 meshIdx 129 repeatIdx   1 primIdx2   2 : PMT_3inch_inner1_solid_ell_helper 
     so_primIdx 3092 numNode    1 nodeOffset 23212 meshIdx 130 repeatIdx   1 primIdx2   3 : PMT_3inch_inner2_solid_ell_helper 
     so_primIdx 3093 numNode    1 nodeOffset 23213 meshIdx 132 repeatIdx   1 primIdx2   4 : PMT_3inch_cntr_solid 



    In [18]: np.unique( st.nds.lvid[st.nds.repeat_index == 2], return_index=True, return_counts=True )
    Out[18]: 
    (array([118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128], dtype=int32),
     array([ 1,  2,  5,  7,  8,  9, 10,  6,  4,  3,  0]),
     array([12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615]))


    In [26]: st.f.factor[:,:4] 
    Out[26]: 
    array([[    0, 25600, 25600,     5],
           [    1, 12615, 12615,    11],
           [    2,  4997,  4997,    14],
           [    3,  2400,  2400,     6],
           [    4,   590,     0,     1],
           [    5,   590,     0,     1],
           [    6,   590,     0,     1],
           [    7,   590,     0,     1],
           [    8,   504,   504,   130]], dtype=int32)





    In [16]: np.unique( st.nds.lvid[st.nds.repeat_index == 2], return_counts=True )
    Out[16]: 
    (array([118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128], dtype=int32),
     array([12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615, 12615]))





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

    print("[--st.descSolids")
    print(st.descSolids(True))
    print(st.descSolids(False))
    print("]--st.descSolids")



