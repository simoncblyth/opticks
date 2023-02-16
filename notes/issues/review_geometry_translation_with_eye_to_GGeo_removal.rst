review_geometry_translation_with_eye_to_GGeo_removal
=======================================================


Comparison of stree.py and CSGFoundry.py python dumping of geometry
------------------------------------------------------------------------

::


    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./sysrap/tests/stree_load_test.sh ana
    epsilon:opticks blyth$ GEOM=J007 RIDX=1 ./CSG/tests/CSGFoundryLoadTest.sh ana


stree_load_test stree::get_transform
--------------------------------------


::

    epsilon:tests blyth$ LVID=102 ./stree_load_test.sh 
    stree::init 
    stree::load_ /tmp/blyth/opticks/U4TreeCreateTest/stree
     LVID 102 num_nds 3
    - ix:  456 dp:    1 sx:    0 pt:  458     nc:    0 fc:   -1 ns:  457 lv:  102     tc:  108 pa:  279 bb:  279 xf:   -1    co
    - ix:  457 dp:    1 sx:    1 pt:  458     nc:    0 fc:   -1 ns:   -1 lv:  102     tc:  105 pa:  280 bb:  280 xf:  169    cy
    - ix:  458 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  456 ns:   -1 lv:  102     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    epsilon:tests blyth$ 



node level semantic transform comparison
--------------------------------------------


After removing the identity suppression in B have one transform for every node::

    In [1]: btr = b.node.view(np.int32)[:,3,3] & 0x7fffffff ; btr 
    Out[2]: array([    1,     2,     3,     4,     5, ..., 25431, 25432, 25433, 25434, 25435], dtype=int32)

    In [3]: np.arange(1,25435+1)
    Out[3]: array([    1,     2,     3,     4,     5, ..., 25431, 25432, 25433, 25434, 25435])

    In [4]: np.all( btr == np.arange(1,25435+1)  )
    Out[4]: True

    In [5]: b.tran.shape
    Out[5]: (25435, 4, 4)

    In [6]: b.node.shape
    Out[6]: (25435, 4, 4)

    In [10]: btran = b.tran[btr-1] ; btran.shape
    Out[10]: (25435, 4, 4)



    In [11]: atr = a.node.view(np.int32)[:,3,3] & 0x7fffffff  ; atr
    Out[11]: array([   1,    2,    3,    0,    4, ..., 8175, 8176, 8177, 8178, 8179], dtype=int32)

    In [12]: a.tran.shape
    Out[12]: (8179, 4, 4)

    In [13]: a.node.shape
    Out[13]: (23547, 4, 4)

    In [14]: atran = a.tran[atr-1] ; atran.shape
    Out[14]: (23547, 4, 4)




::

    In [54]: np.c_[atran[:10], btran[:10]]
    Out[54]: 
    array([[[    1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ],
            [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
            [    0. ,     0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ],
            [    0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ,     1. ]],

           [[    1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ],
            [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
            [    0. ,     0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ],
            [ 3125. ,     0. , 36750. ,     1. ,  3125. ,     0. , 36750. ,     1. ]],

           [[    1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ],
            [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
            [    0. ,     0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ],
            [ 3125. ,     0. , 42250. ,     1. ,  3125. ,     0. , 42250. ,     1. ]],

           [[    1. ,     0. ,     0. ,     0. ,     0. ,     0. ,    -1. ,     0. ],
            [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
            [    0. ,     0. ,     1. ,     0. ,     1. ,     0. ,     0. ,     0. ],
            [    0. ,   831.6,     0. ,     1. ,  3125. ,     0. , 21990. ,     1. ]],



Tran rapidly get out of step::

    In [46]: i0 = 100 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[46]: (0,)

    In [47]: i0 = 0 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[47]: (7,)

    In [48]: i0 = 100 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[48]: (0,)

    In [49]: i0 = 200 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[49]: (81,)

    In [50]: i0 = 300 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[50]: (100,)

    In [51]: i0 = 400 ; i1 = i0+100 ; np.where( np.sum(np.abs(atran[i0:i1]-btran[i0:i1]), axis=(1,2)) > 0 )[0].shape
    Out[51]: (100,)


Hmm doing things at prim level more meaningful, as problems are likely focussed on certain shapes::

    In [64]: np.where( a.pr.numNode != b.pr.numNode )
    Out[64]: (array([2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 3126]),)

    In [66]: np.all( a.pr.nodeOffset[:2375] == b.pr.nodeOffset[:2375] )
    Out[66]: True


first prim with discrep : difference on the transform associated with the union/difference : A looks wrong
----------------------------------------------------------------------------------------------------------------

::

    In [1]: checkprim(a,b,3, True)
    ip:  3 lv:  1/  1 nn:  3/  3 no:  3/  3 mn:                 sTopRock_dome/                 sTopRock_dome 
    Out[1]: 
    ('np.c_[atran, btran], np.c_[atc, aco, btc, bco]',
     array([[[    1. ,     0. ,     0. ,     0. ,     0. ,     0. ,    -1. ,     0. ],
             [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
             [    0. ,     0. ,     1. ,     0. ,     1. ,     0. ,     0. ,     0. ],
             [    0. ,   831.6,     0. ,     1. ,  3125. ,     0. , 21990. ,     1. ]],
     
            [[    0. ,     0. ,    -1. ,     0. ,     0. ,     0. ,    -1. ,     0. ],
             [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
             [    1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ],
             [ 3125. ,     0. , 21990. ,     1. ,  3125. ,     0. , 21990. ,     1. ]],
     
            [[    0. ,     0. ,    -1. ,     0. ,     0. ,     0. ,    -1. ,     0. ],
             [    0. ,     1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ],
             [    1. ,     0. ,     0. ,     0. ,     1. ,     0. ,     0. ,     0. ],
             [ 3125. ,     0. , -7770. ,     1. ,  3125. ,     0. , -7770. ,     1. ]]], dtype=float32),
     array([[  2,   0,   3,   0],
            [105,   0, 105,   0],
            [110,  -1, 110,   0]], dtype=int32))


* ip:4,6,7,10,11,12 all have  same characteristic : difference with root node union/difference tran and A looks wrong 


::


    In [1]: checkprim(a,b,1115)
    ip:1115 lv: 49/ 49 nn:  7/  7 no:6589/6589 mn:GLb1.up02_FlangeI_Web_FlangeII/GLb1.up02_FlangeI_Web_FlangeII tr* 4
    Out[1]: 
    ('np.c_[atran, btran], np.c_[atc, aco, btc, bco], dtran',
     array([[[     1.   ,      0.   ,      0.   ,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.   ,      1.   ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [     0.   ,      0.   ,      1.   ,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [     0.   ,    831.6  ,      0.   ,      1.   ,  -9692.86 , -16788.525,   5852.894,      1.   ]],
     
            [[     1.   ,      0.   ,      0.   ,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.   ,      1.   ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [     0.   ,      0.   ,      1.   ,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [     0.   ,    831.6  ,      0.   ,      1.   ,  -9692.86 , -16788.525,   5852.894,      1.   ]],
     
            [[    -0.145,     -0.25 ,     -0.957,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.866,     -0.5  ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [    -0.479,     -0.829,      0.289,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [ -9600.   , -16627.688,   5796.822,      1.   ,  -9600.   , -16627.688,   5796.822,      1.   ]],
     
            [[    -0.145,     -0.25 ,     -0.957,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.866,     -0.5  ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [    -0.479,     -0.829,      0.289,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [ -9692.86 , -16788.525,   5852.894,      1.   ,  -9692.86 , -16788.525,   5852.894,      1.   ]],
     
            [[    -0.145,     -0.25 ,     -0.957,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.866,     -0.5  ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [    -0.479,     -0.829,      0.289,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [ -9785.721, -16949.363,   5908.966,      1.   ,  -9785.72 , -16949.363,   5908.966,      1.   ]],
     
            [[     1.   ,      0.   ,      0.   ,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.   ,      1.   ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [     0.   ,      0.   ,      1.   ,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [     0.   ,    831.6  ,      0.   ,      1.   ,  -9692.86 , -16788.525,   5852.894,      1.   ]],
     
            [[     1.   ,      0.   ,      0.   ,      0.   ,     -0.145,     -0.25 ,     -0.957,      0.   ],
             [     0.   ,      1.   ,      0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
             [     0.   ,      0.   ,      1.   ,      0.   ,     -0.479,     -0.829,      0.289,      0.   ],
             [     0.   ,    831.6  ,      0.   ,      1.   ,  -9692.86 , -16788.525,   5852.894,      1.   ]]], dtype=float32),
     array([[  1,   0,   1,   0],
            [  1,   0,   1,   0],
            [110,   0, 110,   0],
            [110,   0, 110,   0],
            [110,   0, 110,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]], dtype=int32),
     array([33172.617, 33172.617,     0.   ,     0.   ,     0.001, 33172.617, 33172.617], dtype=float32))







transforms on operator nodes
---------------------------------


In globals the principal difference is with the tranforms on operator nodes.

In locals node that get_ancestors local:true is not stopping before the outer transform::


    ct
    ./CSGFoundryAB.sh 


    In [1]:  checkprim(a,b,3096)                                                                                                                                   
    ip:3096 lv:119/119 nn: 15/ 15 no:23228%24124 tr*15 mn:                          NNVTMCPPMTTail/                          NNVTMCPPMTTail 
    Out[1]: 



    In [2]: ip = 3096

    In [3]: ann = a.pr.numNode[ip]

    In [4]: bnn = b.pr.numNode[ip]

    In [5]: ano = a.pr.nodeOffset[ip]

    In [6]: bno = b.pr.nodeOffset[ip]

    In [7]: atr = ( a.ni.comptran[ano:ano+ann] & 0x7fffffff ) - 1

    In [8]: btr = ( b.ni.comptran[bno:bno+bnn] & 0x7fffffff ) - 1

    In [9]: atc = a.ni.typecode[ano:ano+ann]

    In [10]: btc = b.ni.typecode[bno:bno+bnn]

    In [11]: atc
    Out[11]: array([  2,   1,   2,   1, 105,   2, 105, 103, 105,   0,   0, 103, 105,   0,   0], dtype=int32)

    In [12]: btc
    Out[12]: array([  3,   1,   1,   1, 105,   1, 105, 103, 105,   0,   0, 103, 105,   0,   0], dtype=int32)

    In [13]: atr
    Out[13]: array([  -1,   -1,   -1,   -1, 7961,   -1, 7962, 7963, 7964,   -1,   -1, 7965, 7966,   -1,   -1], dtype=int32)

    In [14]: btr
    Out[14]: array([24124, 24125, 24126, 24127, 24128, 24129, 24130, 24131, 24132, 24133, 24134, 24135, 24136, 24137, 24138], dtype=int32)

    In [15]: a.tran[-1]
    Out[15]: 
    array([[  1. ,   0. ,   0. ,   0. ],
           [  0. ,   1. ,   0. ,   0. ],
           [  0. ,   0. ,   1. ,   0. ],
           [  0. , 831.6,   0. ,   1. ]], dtype=float32)

    In [16]: b.tran[-1]
    Out[16]: 
    array([[  1. ,   0. ,   0. ,   0. ],
           [  0. ,   1. ,   0. ,   0. ],
           [  0. ,   0. ,   1. ,   0. ],
           [  0. , 831.6,   0. ,   1. ]], dtype=float32)





examine the 74/8179 discrepant tran/itra
------------------------------------------

The number of tran/itra is the same between A and B because only leaves have final transforms.
HMM perhaps those transforms actually might be shuffled rather than discrepant ?  
Which would suggest the leaves are being reached in a different order as a result of balancing in A
and not in B. 

::

    ct
    ./CSGFoundryAB.sh


The 2nd stretch of 10 looks like they might be shuffled. And the below confirms that is so

    In [1]: tr = np.arange(8032,8041+1)
    In [5]: np.c_[a.tran[tr],b.tran[tr]]

    In [8]: np.unique(a.tran[tr], axis=0).shape
    Out[8]: (9, 4, 4)

    In [9]: np.unique(b.tran[tr], axis=0).shape
    Out[9]: (9, 4, 4)

    In [10]: np.c_[np.unique(a.tran[tr], axis=0),np.unique(b.tran[tr], axis=0)]
    Out[10]: 
    array([[[   1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
            [-164.   ,    0.   ,  -65.   ,    1.   , -164.   ,    0.   ,  -65.   ,    1.   ]],

           [[   1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
            [-115.966, -115.966,  -65.   ,    1.   , -115.966, -115.966,  -65.   ,    1.   ]],

    In [11]: ub = np.unique(b.tran[tr], axis=0)
    In [12]: ua = np.unique(a.tran[tr], axis=0)

    In [14]: np.all( ua == ub )
    Out[14]: True

    In [16]: np.unique(a.tran[tr], return_index=True, axis=0)[1]
    Out[16]: array([6, 7, 5, 8, 0, 4, 9, 3, 2])

    In [17]: np.unique(b.tran[tr], return_index=True, axis=0)[1]
    Out[17]: array([3, 2, 4, 1, 8, 5, 0, 6, 7])


Check the stretch of 64
---------------------------

::

    In [18]: tr = np.arange(6672,6735+1)   

    In [22]: ub = np.unique(b.tran[tr], axis=0)

    In [23]: ua = np.unique(a.tran[tr], axis=0)

    In [24]: ua.shape
    Out[24]: (56, 4, 4)

    In [25]: ub.shape
    Out[25]: (56, 4, 4)

Doesnt match this time. But its not single prim and not instanced so more involved. ::

    In [30]: checkprims(a,b)
    ip:%(ip)3d lv:%(alv)3d%(slv)s%(blv)3d nn:%(ann)3d%(snn)s%(bnn)3d no:%(ano)3d%(sno)s%(bno)3d tr%(stran)s%(ltran)2d mn:%(amn)40s%(smn)s%(bmn)40s 
    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2382 lv: 93/ 93 nn: 15*127 no:15314%16098 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268 tr*-1 mn:                                    uni1/                                    uni1 

    In [34]: ann=15;bnn=127;ano=15209;bno=15209
    In [35]: anode = a.node[ano:ano+ann]
    In [36]: bnode = b.node[bno:bno+bnn]


    In [39]: atr = anode.view(np.int32)[:,3,3] & 0x7fffffff
    In [40]: btr = bnode.view(np.int32)[:,3,3] & 0x7fffffff

    In [41]: atr
    Out[41]: array([   0,    0,    0,    0,    0,    0,    0, 6673, 6674, 6675, 6676, 6677, 6678, 6679, 6680], dtype=int32)

    In [42]: btr
    Out[42]: 
    array([   0,    0,    0,    0, 6673, 6674, 6675,    0, 6676,    0,    0,    0,    0,    0,    0,    0, 6677,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           6678,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 6679,
           6680,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
          dtype=int32)


    In [43]: at = atr[atr>0] - 1

    In [44]: bt = btr[btr>0] - 1

    In [45]: at               
    Out[45]: array([6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679], dtype=int32)

    In [46]: bt             
    Out[46]: array([6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679], dtype=int32)

    In [53]: ua = np.unique( a.tran[at], axis=0 )
    In [54]: ub = np.unique( b.tran[bt], axis=0 )

    In [59]: np.allclose( ua, ub )
    Out[59]: True


So the transforms of prim 2375 are confirmed to match but they are shuffled between A and B::

    In [60]: np.unique( a.tran[at], return_index=True, axis=0 )[1]
    Out[60]: array([2, 1, 4, 0, 6, 7, 3])

    In [61]: np.unique( b.tran[bt], return_index=True, axis=0 )[1]
    Out[61]: array([3, 4, 6, 5, 1, 2, 0])


::

    In [2]: checkprims(a,b,2375,2382)
    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971 tr* 8 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 

    In [4]: checkprims(a,b,-1,-1,order="S")
    ip:%(ip)4d lv:%(alv)3d%(slv)s%(blv)3d nn:%(ann)3d%(snn)s%(bnn)3d no:%(ano)5d%(sno)s%(bno)5d      S tr%(stran)s%(ltran)2d mn:%(amn)40s%(smn)s%(bmn)40s 
    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2382 lv: 93/ 93 nn: 15*127 no:15314%16098      S tr: 0 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 

    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268      S tr* 6 mn:                                    uni1/                                    uni1 

                    ## ordering by sum of 16 elements does not establish a reliable order for transforms that 
                    ## are arranged symmetrically : but using unique shows that the transforms are shuffled

    In [7]: checkprims(a,b,3126,3127,order="U")
    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268      U tr: 0 mn:                                    uni1/                                    uni1 


Have confirmed that all the transforms match.  BUT LV 93:solidSJReceiverFastern and 99:uni1 differ in that A is balanced, but not B.
Somehow the balancing is resulting in the ordering of the primitives changing.  



transform comparison after elliposoid stomp avoidance
--------------------------------------------------------

CSGFoundryAB.sh down to 74/8179 discrepant tran/itra that are tangled with lack of tree balancing for lvid 93:solidSJReceiverFastern 99:uni1

::

    In [40]: a.tran.shape, b.tran.shape
    Out[40]: ((8179, 4, 4), (8179, 4, 4))


    In [2]: sab = np.sum(np.abs(a.tran-b.tran), axis=(1,2) )
    In [3]: vab = np.sum(np.abs(a.itra-b.itra), axis=(1,2) )

    In [5]: np.c_[a.tran, b.tran][np.where(sab > 0.05)].shape
    Out[5]: (74, 4, 8)

    In [6]: w = np.where(sab > 0.05)[0] ; w 
    array([6672, 6673, 6674, 6675, 6676, 6677, 6678, 6679, 6680, 6681, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6691, 6692, 6693, 6694, 6695, 6696, 6697, 6698, 6699, 6700, 6701, 6702, 6703,
           6704, 6705, 6706, 6707, 6708, 6709, 6710, 6711, 6712, 6713, 6714, 6715, 6716, 6717, 6718, 6719, 6720, 6721, 6722, 6723, 6724, 6725, 6726, 6727, 6728, 6729, 6730, 6731, 6732, 6733, 6734, 6735,
           8032, 8033, 8034, 8035, 8036, 8037, 8038, 8039, 8040, 8041])


    In [13]: np.all( w == np.concatenate( [np.arange(6672,6735+1), np.arange(8032,8041+1)] )  )
    Out[13]: True

    ## two contiguous stretches of transforms are discrepant 

    In [15]: np.arange(6672,6735+1).shape
    Out[15]: (64,)

    In [16]: np.arange(8032,8041+1).shape
    Out[16]: (10,)

Search for the prim those transform pointers correspond to::

    In [29]: b.pr.tranOffset[2375:2385]
    Out[29]: array([6672, 6680, 6688, 6696, 6704, 6712, 6720, 6728, 6736, 6737], dtype=int32)

    In [30]: a.pr.tranOffset[2375:2385]
    Out[30]: array([6672, 6680, 6688, 6696, 6704, 6712, 6720, 6728, 6736, 6737], dtype=int32)


    In [37]: a.pr.tranOffset[3126:3128]
    Out[37]: array([8032, 8042], dtype=int32)

    In [38]: b.pr.tranOffset[3126:3128]
    Out[38]: array([8032, 8042], dtype=int32)


Those prim correspond to what checkprims gives::

    In [39]: checkprims(a,b)
    ip:%(ip)3d lv:%(alv)3d%(slv)s%(blv)3d nn:%(ann)3d%(snn)s%(bnn)3d no:%(ano)3d%(sno)s%(bno)3d tr%(stran)s%(ltran)2d mn:%(amn)40s%(smn)s%(bmn)40s 

    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2382 lv: 93/ 93 nn: 15*127 no:15314%16098 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 

    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268 tr*-1 mn:                                    uni1/                                    uni1 


So the remaining 74 discrepant tran/itra need tree balancing first. 



change to leaf only final transforms in CSGImport::importNode
----------------------------------------------------------------

* makes the tran/itra counts the same 

::

    ct  
    ./CSGFoundryAB.sh 

    In [16]: a.tran.shape
    Out[16]: (8179, 4, 4)

    In [17]: b.tran.shape
    Out[17]: (8179, 4, 4)
    

    In [9]: sab = np.sum(np.abs(a.tran-b.tran), axis=(1,2) ) 

    In [14]: w = np.where(sab > 0.001 )[0] ; w 
    Out[15]: array([ 422,  425,  428,  431,  434, ..., 8037, 8038, 8039, 8040, 8041])
        

    In [20]: np.c_[a.tran, b.tran][np.where(sab > 0.05)].shape      ## big epsilon to avoid float/double diffs
    Out[20]: (79, 4, 8)     # 79/8179 with significant diffs  (5 from ellipsoid stomp?)


    In [22]: vab = np.sum(np.abs(a.itra-b.itra), axis=(1,2) )
    In [25]: np.c_[a.itra, b.itra][np.where(vab > 0.05)].shape
    Out[25]: (79, 4, 8)

    In [5]: np.c_[a.tran, b.tran][np.where(sab > 0.05)].shape   ## down 5 after avoid the stomp
    Out[5]: (74, 4, 8)


    In [29]: np.all( np.where( sab > 0.05 )[0] == np.where( vab > 0.05 )[0] )
    Out[29]: True





    In [21]: np.c_[a.tran, b.tran][np.where(sab > 0.05)]
    Out[21]: 
    array([[[     0.   ,     -0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ],
            [    -0.5  ,     -0.866,      0.   ,      0.   ,     -0.5  ,     -0.866,      0.   ,      0.   ],
            [     0.866,     -0.5  ,     -0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
            [-15314.271,   8841.699,     52.   ,      1.   , -15289.271,   8885.   ,      0.   ,      1.   ]],

           [[     0.   ,     -0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ],
            [    -0.5  ,     -0.866,      0.   ,      0.   ,     -0.5  ,     -0.866,      0.   ,      0.   ],
            [     0.866,     -0.5  ,     -0.   ,      0.   ,      0.866,     -0.5  ,      0.   ,      0.   ],
            [-15314.271,   8841.699,    -52.   ,      1.   , -15304.312,   8835.949,      0.   ,      1.   ]],
      
           ...

           [[     1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ],
            [    -0.   ,   -164.   ,    -65.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ]],

           [[     1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ,      0.   ],
            [   115.966,   -115.966,    -65.   ,      1.   ,      0.   ,      0.   ,      0.   ,      1.   ]]], dtype=float32)

    In [22]:                                                                           



::

    In [30]: checkprims(a,b)
    ip:%(ip)3d lv:%(alv)3d%(slv)s%(blv)3d nn:%(ann)3d%(snn)s%(bnn)3d no:%(ano)3d%(sno)s%(bno)3d tr%(stran)s%(ltran)2d mn:%(amn)40s%(smn)s%(bmn)40s 

    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2382 lv: 93/ 93 nn: 15*127 no:15314%16098 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 

    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268 tr*-1 mn:                                    uni1/                                    uni1 

    Above two from balancing vs not  




    ip:3107 lv:105/105 nn: 15/ 15 no:23271%24167 tr* 2 mn:                     HamamatsuR12860Tail/                     HamamatsuR12860Tail 
    ip:3108 lv:116/116 nn: 15/ 15 no:23286%24182 tr* 1 mn:HamamatsuR12860_PMT_20inch_pmt_solid_1_4/HamamatsuR12860_PMT_20inch_pmt_solid_1_4 
    ip:3109 lv:115/115 nn: 15/ 15 no:23301%24197 tr* 1 mn:HamamatsuR12860_PMT_20inch_body_solid_1_4/HamamatsuR12860_PMT_20inch_body_solid_1_4 
    ip:3111 lv:114/114 nn:  7/  7 no:23317%24213 tr* 1 mn:HamamatsuR12860_PMT_20inch_inner2_solid_1_4/HamamatsuR12860_PMT_20inch_inner2_solid_1_4 


Mismatched transforms in prim 3107,3108,3109,3111 are all from the translation stomping on an ellipsoid scale transform::

    In [5]: checkprim(a,b,3108)
    ip:3108 lv:116/116 nn: 15/ 15 no:23286%24182 tr* 1 mn:HamamatsuR12860_PMT_20inch_pmt_solid_1_4/HamamatsuR12860_PMT_20inch_pmt_solid_1_4 
    Out[5]: 
    ('np.c_[atran, btran], np.c_[atc, aco, btc, bco], dtran',
     array([[[   1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ],
             [   0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
             [   0.   ,    0.   , -179.216,    1.   ,    0.   ,    0.   , -179.216,    1.   ]],
     
            [[   1.337,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ],
             [   0.   ,    1.337,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
             [   0.   ,    0.   ,   -5.   ,    1.   ,    0.   ,    0.   ,   -5.   ,    1.   ]],
     
            [[   1.337,    0.   ,    0.   ,    0.   ,    1.337,    0.   ,    0.   ,    0.   ],
             [   0.   ,    1.337,    0.   ,    0.   ,    0.   ,    1.337,    0.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
             [   0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ]],
     
            [[   1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ],
             [   0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ,    0.   ,    0.   ,    1.   ,    0.   ],
             [   0.   ,    0.   ,   -2.5  ,    1.   ,    0.   ,    0.   ,   -2.5  ,    1.   ]]], dtype=float32),
     array([[  1,   0,   1,   0],
            [  1,   0,   1,   0],
            [108,   0, 108,   0],
            [  1,   0,   1,   0],
            [103,   0, 103,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [103,   0, 103,   0],
            [105,   0, 105,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]], dtype=int32),
     array([0.   , 0.674, 0.   , 0.   ], dtype=float32))


prevent ellipsoid stomping
-----------------------------

::

    ./U4TreeCreateTest.sh 

    [ U4Tree::Create 
    snd::setXF STOMPING xform 182

     t                                                      v                                                      t*v                                                   
     1.0000     0.0000     0.0000     0.0000                1.0000     -0.0000    0.0000     -0.0000               1.0000     0.0000     0.0000     0.0000    
     0.0000     1.0000     0.0000     0.0000                -0.0000    1.0000     -0.0000    0.0000                0.0000     1.0000     0.0000     0.0000    
     0.0000     0.0000     1.0000     0.0000                0.0000     -0.0000    1.0000     -0.0000               0.0000     0.0000     1.0000     0.0000    
     0.0000     0.0000     -5.0000    1.0000                -0.0000    0.0000     5.0000     1.0000                0.0000     0.0000     0.0000     1.0000    

    snd::setXF STOMPING xform 185


After avoiding the stomping::

    In [1]: checkprims(a,b)
    ip:%(ip)3d lv:%(alv)3d%(slv)s%(blv)3d nn:%(ann)3d%(snn)s%(bnn)3d no:%(ano)3d%(sno)s%(bno)3d tr%(stran)s%(ltran)2d mn:%(amn)40s%(smn)s%(bmn)40s 
    ip:2375 lv: 93/ 93 nn: 15*127 no:15209/15209 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2376 lv: 93/ 93 nn: 15*127 no:15224%15336 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2377 lv: 93/ 93 nn: 15*127 no:15239%15463 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2378 lv: 93/ 93 nn: 15*127 no:15254%15590 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2379 lv: 93/ 93 nn: 15*127 no:15269%15717 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2380 lv: 93/ 93 nn: 15*127 no:15284%15844 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2381 lv: 93/ 93 nn: 15*127 no:15299%15971 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:2382 lv: 93/ 93 nn: 15*127 no:15314%16098 tr*-1 mn:                  solidSJReceiverFastern/                  solidSJReceiverFastern 
    ip:3126 lv: 99/ 99 nn: 31*1023 no:23372%24268 tr*-1 mn:                                    uni1/                                    uni1 

    In [2]:                       






in old geometry the operator nodes are referencing a non-sensical gtransform : but its never used 
------------------------------------------------------------------------------------------------------------------------

Formerly thought had nonsensical operator node transforms : that was just the "-1" pointing to the last transform


**Operator nodes pick between intersect distances from their leaf nodes, they never use their own gtransforms.**

This means can substantially reduce the size of the tran/itra buffers as only leaf nodes need to 
reference their transforms. For clarity can set the operator node transform references to zero
meaning no associated transform. 

gtransformIdx grepping is consistent with this::

    epsilon:CSG blyth$ grep gtransformIdx *.h 
    CSGNode.h:    |    |                |                |  typecode      | gtransformIdx  |                                                 |
    CSGNode.h:    NODE_METHOD void setComplement( bool complement ){  setTransformComplement( gtransformIdx(), complement) ; }
    CSGNode.h:    NODE_METHOD unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
    csg_intersect_leaf.h:    const unsigned gtransformIdx = node->gtransformIdx() ; 
    csg_intersect_leaf.h:    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    csg_intersect_leaf.h:    const unsigned gtransformIdx = node->gtransformIdx() ; 
    csg_intersect_leaf.h:    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None
    csg_intersect_leaf.h:    printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    csg_intersect_leaf.h:    //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    epsilon:CSG blyth$ 



importTree
--------------

A: old GGeo workflow has 8179 tran/itra for 23547 node 
B: CSGImport has 25423 tran/itra for 25435 node

Note that adding nullptr adds an identity transform, but 
identities cannot explain the big difference as there are 
only ~36 nodes with identity transforms. 

Looks like there is suppression of the same transforms
done at GGeo/GTransform level::

    In [59]: u_bt, n_bt = np.unique( B.tran, return_counts=True, axis=0 ) ; u_bt.shape
    Out[60]: (7931, 4, 4)

    In [64]: u_at, n_at = np.unique( A.tran, return_counts=True, axis=0 ) ; u_at.shape
    Out[65]: (7946, 4, 4)

After uniquing the count is close. But note the uniqing looks to 
have been done within each compound, as there are still duplicate transforms 
in A, just much less than B::

    In [71]: n_bt[n_bt > 1 ].shape
    Out[71]: (2446,)

    In [72]: n_at[n_at > 1 ].shape 
    Out[72]: (151,)

How to check within the original pools ? Use CSGPrim to see which tranOffset correspond to each ridx::

    In [4]: a.pr.tranOffset[a.pr.repeatIdx == 0]
    Out[4]: array([   0,    1,    2,    3,    5, ..., 7933, 7942, 7943, 7945, 7946], dtype=int32)

    In [5]: a.pr.tranOffset[a.pr.repeatIdx == 1]
    Out[5]: array([7948, 7950, 7951, 7952, 7953], dtype=int32)

    In [6]: a.pr.tranOffset[a.pr.repeatIdx == 2]
    Out[6]: array([7954, 7957, 7961, 7967, 7968, 7969, 7970, 7971, 7973, 7975, 7977], dtype=int32)

    In [7]: a.pr.tranOffset[a.pr.repeatIdx == 3]
    Out[7]: array([7978, 7981, 7985, 7991, 7995, 7999, 8000, 8003, 8005, 8007, 8009, 8011, 8013, 8014], dtype=int32)

    In [8]: a.pr.tranOffset[a.pr.repeatIdx == 4]
    Out[8]: array([8016, 8017, 8021, 8023, 8025, 8028], dtype=int32)

    In [12]: a.pr.tranOffset[a.pr.repeatIdx == 5]
    Out[12]: array([8031], dtype=int32)

    In [13]: a.pr.tranOffset[a.pr.repeatIdx == 6]
    Out[13]: array([8032], dtype=int32)

    In [14]: a.pr.tranOffset[a.pr.repeatIdx == 7]
    Out[14]: array([8042], dtype=int32)

    In [15]: a.pr.tranOffset[a.pr.repeatIdx == 8]
    Out[15]: array([8046], dtype=int32)

    In [16]: a.pr.tranOffset[a.pr.repeatIdx == 9]
    Out[16]: 
    array([8049, 8050, 8051, 8052, 8053, 8054, 8055, 8056, 8057, 8058, 8059, 8060, 8061, 8062, 8063, 8064, 8065, 8066, 8067, 8068, 8069, 8070, 8071, 8072, 8073, 8074, 8075, 8076, 8077, 8078, 8079, 8080,
           8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089, 8090, 8091, 8092, 8093, 8094, 8095, 8096, 8097, 8098, 8099, 8100, 8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112,
           8113, 8114, 8115, 8116, 8117, 8118, 8119, 8120, 8121, 8122, 8123, 8124, 8125, 8126, 8127, 8128, 8129, 8130, 8131, 8132, 8133, 8134, 8135, 8136, 8137, 8138, 8139, 8140, 8141, 8142, 8143, 8144,
           8145, 8146, 8147, 8148, 8149, 8150, 8151, 8152, 8153, 8154, 8155, 8156, 8157, 8158, 8159, 8160, 8161, 8162, 8163, 8164, 8165, 8166, 8167, 8168, 8169, 8170, 8171, 8172, 8173, 8174, 8175, 8176,
           8177, 8178], dtype=int32)

    In [17]: a.tran.shape
    Out[17]: (8179, 4, 4)




All ridx 9 tran appear twice::

    In [20]: np.all( np.unique(a.tran[8049:], return_counts=True, axis=0 )[1] == 2 )
    Out[20]: True

Look like next to each other::

    In [22]: a.tran[8049:]
    Out[22]: 
    array([[[   1. ,    0. ,    0. ,    0. ],
            [   0. ,    1. ,    0. ,    0. ],
            [   0. ,    0. ,    1. ,    0. ],
            [   0. ,    0. ,    0. ,    1. ]],

           [[   1. ,    0. ,    0. ,    0. ],
            [   0. ,    1. ,    0. ,    0. ],
            [   0. ,    0. ,    1. ,    0. ],
            [   0. ,    0. ,    0. ,    1. ]],

           [[   1. ,    0. ,    0. ,    0. ],
            [   0. ,    1. ,    0. ,    0. ],
            [   0. ,    0. ,    1. ,    0. ],
            [   0. , -831.6,    0. ,    1. ]],

           [[   1. ,    0. ,    0. ,    0. ],
            [   0. ,    1. ,    0. ,    0. ],
            [   0. ,    0. ,    1. ,    0. ],
            [   0. , -831.6,    0. ,    1. ]],



Hmm still quite some duplication of transforms::

    In [23]: a.tran[7978:8016]
    Out[23]: 
    array([[[   1.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    1.   ]],

           [[   1.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    1.   ]],

           [[   1.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    1.   ]],

           [[   1.32 ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    1.32 ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    1.   ]],



Old workflow has some level of repeated transform suppression, 
but its far from perfect which makes it difficult to reproduce. 

Also at some stage decided that every CSG node should get a transform.
Maybe the duplicate transform suppression just operated at structural 
level ?  

Actually checking in some old geocache see that the duplicate suppression 
was active but far from perfect (maybe float precision effect). 

HMM: probably easiest to compare at node level by derefing the transform
rather than compare at transform level 

So that means effectively forming a transform for every node.

::

    In [4]: btr = b.node.view(np.int32)[:,3,3] & 0x7fffffff

    In [5]: atr = a.node.view(np.int32)[:,3,3] & 0x7fffffff

    In [6]: atr
    Out[6]: array([   1,    2,    3,    0,    4, ..., 8175, 8176, 8177, 8178, 8179], dtype=int32)

    In [7]: btr
    Out[7]: array([    0,     1,     2,     3,     4, ..., 25419, 25420, 25421, 25422, 25423], dtype=int32)



Transform uniquing
--------------------


::

    220 /**
    221 NCSGData::addUniqueTransform
    222 ------------------------------
    223 
    224 Used global transforms are collected into the GTransforms
    225 buffer and the 1-based index to the transforms is returned. 
    226 This is invoked from NCSG::addUniqueTransform
    227 
    228 **/
    230 unsigned NCSGData::addUniqueTransform( const nmat4triple* gtransform )
    231 {
    232     NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
    233     gtmp->zero();
    234     gtmp->setMat4Triple( gtransform, 0);
    235 
    236     NPY<float>* gtransforms = getGTransformBuffer();
    237     assert(gtransforms);
    238     unsigned gtransform_idx = 1 + gtransforms->addItemUnique( gtmp, 0 ) ;
    239     delete gtmp ;
    240 
    241     return gtransform_idx ;
    242 }

    1081 unsigned NCSG::addUniqueTransform( const nmat4triple* gtransform_ )
    1082 {
    1083     bool no_offset = m_gpuoffset.x == 0.f && m_gpuoffset.y == 0.f && m_gpuoffset.z == 0.f ;
    1084 
    1085     bool reverse = true ; // <-- apply transfrom at root of transform hierarchy (rather than leaf)
    1086 
    1087     bool match = true ;
    1088 
    1089     const nmat4triple* gtransform = no_offset ? gtransform_ : gtransform_->make_translated(m_gpuoffset, reverse, "NCSG::addUniqueTransform", match ) ;
    1090 
    1091     if(!match)
    1092     {
    1093         LOG(error) << "matrix inversion precision issue ?" ;
    1094     }
    1095 
    1096     /*
    1097     std::cout << "NCSG::addUniqueTransform"
    1098               << " orig " << *gtransform_
    1099               << " tlated " << *gtransform
    1100               << " gpuoffset " << m_gpuoffset 
    1101               << std::endl 
    1102               ;
    1103     */
    1104     return m_csgdata->addUniqueTransform( gtransform );   // add to m_gtransforms
    1105 }




Getting transforms in instanced case
----------------------------------------

::

    In [16]: cf.meshname[118]
    Out[16]: 'NNVTMCPPMTsMask'

    In [17]: ln = st.find_lvid_nodes(118)
    In [18]: ln.shape
    Out[18]: (12615,)

    In [22]: np.all( np.unique(ln, return_counts=True)[1] == 1 )  ## all unique as expected
    Out[22]: True

    In [37]: st.f.m2w[ln].reshape(-1,16)   ## HMM LOOK TO BE ALL IDENTITY 
    Out[37]: 
    array([[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],

    In [38]: snode.Type()
    snode.Type()
      0 :             snode.ix : index 
      1 :             snode.dp : depth 
      2 :             snode.sx : sibdex 
      3 :             snode.pt : parent 
      4 :             snode.nc : num_child 
      5 :             snode.fc : first_child 
      6 :             snode.sx : next_sibling 
      7 :             snode.lv : lvid 
      8 :             snode.cp : copyno 
      9 :             snode.se : sensor_id 
     10 :             snode.sx : sensor_index 
     11 :             snode.ri : repeat_index 
     12 :             snode.ro : repeat_ordinal 
     13 :             snode.bd : boundary 
              ix     dp     sx     pt     nc     fc     sx     lv     cp     se     sx     ri     ro     bd.Label() : 


    In [36]: snode.Label(6,11), st.f.nds[ln]
    Out[36]: 
    ('           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      ro      bd',
     array([[ 70994,      7,      0,  70993,      0,     -1,  70995,    118,      0,     -1,     -1,      2,      0,     27],
            [ 71019,      7,      0,  71018,      0,     -1,  71020,    118,      0,     -1,     -1,      2,      1,     27],
            [ 71044,      7,      0,  71043,      0,     -1,  71045,    118,      0,     -1,     -1,      2,      2,     27],
            [ 71251,      7,      0,  71250,      0,     -1,  71252,    118,      0,     -1,     -1,      2,      3,     27],
            [ 71262,      7,      0,  71261,      0,     -1,  71263,    118,      0,     -1,     -1,      2,      4,     27],
            [ 71273,      7,      0,  71272,      0,     -1,  71274,    118,      0,     -1,     -1,      2,      5,     27],
            [ 71284,      7,      0,  71283,      0,     -1,  71285,    118,      0,     -1,     -1,      2,      6,     27],
            [ 71295,      7,      0,  71294,      0,     -1,  71296,    118,      0,     -1,     -1,      2,      7,     27],
            [ 71306,      7,      0,  71305,      0,     -1,  71307,    118,      0,     -1,     -1,      2,      8,     27],
            ...
            [279329,      7,      0, 279328,      0,     -1, 279330,    118,      0,     -1,     -1,      2,  12609,     27],
            [279340,      7,      0, 279339,      0,     -1, 279341,    118,      0,     -1,     -1,      2,  12610,     27],
            [279351,      7,      0, 279350,      0,     -1, 279352,    118,      0,     -1,     -1,      2,  12611,     27],
            [279362,      7,      0, 279361,      0,     -1, 279363,    118,      0,     -1,     -1,      2,  12612,     27],
            [279373,      7,      0, 279372,      0,     -1, 279374,    118,      0,     -1,     -1,      2,  12613,     27],
            [279384,      7,      0, 279383,      0,     -1, 279385,    118,      0,     -1,     -1,      2,  12614,     27]], dtype=int32))

    In [39]: st.get_ancestors(70994)
    Out[39]: [0, 65722, 65723, 65724, 67845, 67846, 70993]

    In [40]: anc = st.get_ancestors(70994)

    In [41]: snode.Label(6,11), st.f.nds[anc]
    Out[41]: 
    ('           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      ro      bd',
     array([[     0,      0,     -1,     -1,      2,      1,     -1,    149,      0,     -1,     -1,      0,     -1,      0],
            [ 65722,      1,      1,      0,      1,  65723,     -1,    148,      0,     -1,     -1,      0,     -1,      1],
            [ 65723,      2,      0,  65722,      1,  65724,     -1,    147,      0,     -1,     -1,      0,     -1,     13],
            [ 65724,      3,      0,  65723,   4521,  65725,     -1,    146,      0,     -1,     -1,      0,     -1,     14],
            [ 67845,      4,   2120,  65724,      1,  67846, 407692,    139,      0,     -1,     -1,      0,     -1,     16],
            [ 67846,      5,      0,  67845,  46276,  67847,     -1,    138,      0,     -1,     -1,      0,     -1,     17],
            [ 70993,      6,   3065,  67846,      3,  70994,  71004,    128,      2,      2,    506,      2,      0,     26]], dtype=int32))

    In [42]: cf.meshname[128]
    Out[42]: 'NNVTMCPPMTsMask_virtual'

    In [43]: ln2 = st.find_lvid_nodes(128)

    In [44]: ln2.shape
    Out[44]: (12615,)

    In [45]: ln2
    Out[45]: 
    array([ 70993,  71018,  71043,  71250,  71261,  71272,  71283,  71294,  71305,  71316,  71327,  71338,  71349,  71360,  71371,  71382, ..., 279218, 279229, 279240, 279251, 279262, 279273, 279284,
           279295, 279306, 279317, 279328, 279339, 279350, 279361, 279372, 279383])


    In [52]: np.set_printoptions(linewidth=250)

    In [53]: st.f.m2w[ln2].reshape(-1,16)
    Out[53]: 
    array([[    -1.   ,      0.   ,     -0.   ,      0.   ,      0.   ,      1.   ,     -0.   ,      0.   ,      0.   ,      0.   ,     -1.   ,      0.   ,    316.078,   -882.079,  19365.   ,      1.   ],
           [    -1.   ,      0.   ,     -0.   ,      0.   ,      0.   ,      1.   ,     -0.   ,      0.   ,      0.   ,      0.   ,     -1.   ,      0.   ,    789.63 ,    504.435,  19365.   ,      1.   ],
           [    -1.   ,      0.   ,     -0.   ,      0.   ,      0.   ,      1.   ,     -0.   ,      0.   ,      0.   ,      0.   ,     -1.   ,      0.   ,   -667.496,    657.585,  19365.   ,      1.   ],
           [     0.98 ,      0.17 ,      0.099,      0.   ,      0.171,     -0.985,      0.   ,      0.   ,      0.098,      0.017,     -0.995,      0.   ,  -1899.448,   -328.712,  19338.16 ,      1.   ],
           [     0.893,      0.439,      0.099,      0.   ,      0.441,     -0.897,      0.   ,      0.   ,      0.089,      0.044,     -0.995,      0.   ,  -1729.898,   -850.533,  19338.16 ,      1.   ],
           [     0.733,      0.673,      0.099,      0.   ,      0.676,     -0.737,      0.   ,      0.   ,      0.073,      0.067,     -0.995,      0.   ,  -1420.202,  -1303.449,  19338.16 ,      1.   ],
           [     0.413,      0.905,      0.099,      0.   ,      0.91 ,     -0.415,      0.   ,      0.   ,      0.041,      0.09 ,     -0.995,      0.   ,   -800.787,  -1753.48 ,  19338.16 ,      1.   ]


HMM: need to identify the ridx and the containing node in order to know where the base transform is

::

    In [57]: st.f.inst_nidx.shape                                                                                                                                
    Out[57]: (48477,)

    In [59]: np.where( st.f.inst_nidx == ln2[0] )[0]                                                                                                             
    Out[59]: array([25601])

    In [60]: np.where( st.f.inst_nidx == ln2[1] )[0]                                                                                                             
    Out[60]: array([25602])

    In [61]: np.where( st.f.inst_nidx == ln2[2] )[0]                                                                                                             
    Out[61]: array([25603])


    In [63]: st.f.inst[25601]                                                                                                                                    
    Out[63]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [  316.078,  -882.079, 19365.   ,     0.   ]])

    In [64]: st.f.m2w[ln2[0]]                                                                                                                                    
    Out[64]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,    -0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [  316.078,  -882.079, 19365.   ,     1.   ]])

    In [65]: st.f.inst[25602]                                                                                                                                    
    Out[65]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,     0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [  789.63 ,   504.435, 19365.   ,     0.   ]])

    In [66]: st.f.m2w[ln2[1]]                                                                                                                                    
    Out[66]: 
    array([[   -1.   ,     0.   ,    -0.   ,     0.   ],
           [    0.   ,     1.   ,    -0.   ,     0.   ],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [  789.63 ,   504.435, 19365.   ,     1.   ]])


    # first 3089 prim are for remainder ridx 0                                                                          
    In [2]: CSGPrim.Label(), cf.prim.view(np.int32)[:,:2].reshape(-1,8)[:3089]   
    Out[2]: 
    ('          nn     no     to     po     sb     lv     ri     pi',
     array([[    1,     0,     0,     0,     0,   149,     0,     0],
            [    1,     1,     1,     0,     1,    17,     0,     1],
            [    1,     2,     2,     0,     2,     2,     0,     2],
            [    3,     3,     3,     0,     3,     1,     0,     3],
            [    3,     6,     5,     0,     4,     0,     0,     4],
            ...,
            [  127, 22818,  7915,     0,  3082,   103,     0,  3082],
            [  127, 22945,  7924,     0,  3083,   103,     0,  3083],
            [  127, 23072,  7933,     0,  3084,   103,     0,  3084],
            [    1, 23199,  7942,     0,  3085,   137,     0,  3085],
            [    3, 23200,  7943,     0,  3086,   134,     0,  3086],
            [    1, 23203,  7945,     0,  3087,   135,     0,  3087],
            [    3, 23204,  7946,     0,  3088,   136,     0,  3088]], dtype=int32))


    # last 170 prim are for the instances 
    In [93]: cf.prim.view(np.int32)[:,:2].reshape(-1,8)[3089:].shape
    Out[93]: (170, 8)


    In [3]: CSGPrim.Label(), cf.prim.view(np.int32)[:,:2].reshape(-1,8)[3089:]
    Out[3]: 
    ('          nn     no     to     po     sb     lv     ri     pi',
     array([[    3, 23207,  7948,     0,     0,   133,     1,     0],
            [    1, 23210,  7950,     0,     1,   131,     1,     1],
            [    1, 23211,  7951,     0,     2,   129,     1,     2],
            [    1, 23212,  7952,     0,     3,   130,     1,     3],
            [    1, 23213,  7953,     0,     4,   132,     1,     4],

            [    7, 23214,  7954,     0,     0,   128,     2,     0],
            [    7, 23221,  7957,     0,     1,   118,     2,     1],
            [   15, 23228,  7961,     0,     2,   119,     2,     2],
            [    1, 23243,  7967,     0,     3,   127,     2,     3],
            ...,
            [    1, 23542,  8174,     0,   125,     8,     9,   125],
            [    1, 23543,  8175,     0,   126,     9,     9,   126],
            [    1, 23544,  8176,     0,   127,     8,     9,   127],
            [    1, 23545,  8177,     0,   128,     9,     9,   128],
            [    1, 23546,  8178,     0,   129,     8,     9,   129]], dtype=int32))


    In [78]: cf.node.shape                                                                                                                                       
    Out[78]: (23547, 4, 4)


    In [9]: CSGPrim.Label(), cf.prim[np.where( cf.pr.repeatIdx == 1 )].view(np.int32)[:,:2].reshape(-1,8)
    Out[9]: 
    ('          nn     no     to     po     sb     lv     ri     pi',
     array([[    3, 23207,  7948,     0,     0,   133,     1,     0],
            [    1, 23210,  7950,     0,     1,   131,     1,     1],
            [    1, 23211,  7951,     0,     2,   129,     1,     2],
            [    1, 23212,  7952,     0,     3,   130,     1,     3],
            [    1, 23213,  7953,     0,     4,   132,     1,     4]], dtype=int32))



    In [12]: cf.tran[7948:7954].reshape(-1,16)                                                                                                               
    Out[12]: 
    array([[1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
           [1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
           [1.667, 0.   , 0.   , 0.   , 0.   , 1.667, 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
           [1.727, 0.   , 0.   , 0.   , 0.   , 1.727, 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
           [1.727, 0.   , 0.   , 0.   , 0.   , 1.727, 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
           [1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 1.   ]], dtype=float32)

    In [10]: CSGPrim.Label(), cf.prim[np.where( cf.pr.repeatIdx == 2 )].view(np.int32)[:,:2].reshape(-1,8)
    Out[10]: 
    ('          nn     no     to     po     sb     lv     ri     pi',
     array([[    7, 23214,  7954,     0,     0,   128,     2,     0],
            [    7, 23221,  7957,     0,     1,   118,     2,     1],
            [   15, 23228,  7961,     0,     2,   119,     2,     2],
            [    1, 23243,  7967,     0,     3,   127,     2,     3],
            [    1, 23244,  7968,     0,     4,   126,     2,     4],
            [    1, 23245,  7969,     0,     5,   120,     2,     5],
            [    1, 23246,  7970,     0,     6,   125,     2,     6],
            [    3, 23247,  7971,     0,     7,   121,     2,     7],
            [    3, 23250,  7973,     0,     8,   122,     2,     8],
            [    3, 23253,  7975,     0,     9,   123,     2,     9],
            [    1, 23256,  7977,     0,    10,   124,     2,    10]], dtype=int32))


    In [57]: snd.Label(3,8),st.f.csg.node[st.csg.lvid == 128],st.soname_[128]                                                                    
    Out[57]: 
    ('        ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf',
     array([[575,   1,   0, 578,   0,  -1, 576, 128, 105, 349, 349,  -1,   0,   0,   0,   0],
            [576,   1,   1, 578,   0,  -1, 577, 128, 105, 350, 350,  -1,   0,   0,   0,   0],
            [577,   1,   2, 578,   0,  -1,  -1, 128, 108, 351, 351,  -1,   0,   0,   0,   0],
            [578,   0,  -1,  -1,   3, 575,  -1, 128,  11,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32),
     b'NNVTMCPPMTsMask_virtual')           ## CSG_CONTIGUOUS : no transforms


    In [81]: cf.meshname[(128,118,119,127,126,120,125,121,122,123,124),]                                                                         
    Out[81]: 
    array(['NNVTMCPPMTsMask_virtual', 
           'NNVTMCPPMTsMask', 
           'NNVTMCPPMTTail', 
           'NNVTMCPPMT_PMT_20inch_pmt_solid_head', 
           'NNVTMCPPMT_PMT_20inch_body_solid_head', 
           'NNVTMCPPMT_PMT_20inch_inner1_solid_head',
           'NNVTMCPPMT_PMT_20inch_inner2_solid_head', 
           'NNVTMCPPMT_PMT_20inch_edge_solid', 
           'NNVTMCPPMT_PMT_20inch_plate_solid', 
           'NNVTMCPPMT_PMT_20inch_tube_solid', 
           'NNVTMCPPMT_PMT_20inch_mcp_solid'],
          dtype=object)

    In [66]: ntc = cf.node.view(np.int32)[:,3,2]
    In [67]: ntr = cf.node.view(np.int32)[:,3,3] & 0x7fffffff

    In [69]: ntc[23221:23221+7]                                      ## CSG nodes of lv 118
    Out[69]: array([  2,   1,   2, 103, 105, 103, 105], dtype=int32)
                  ## in   un   in   zs   cy   zs   cy

    In [70]: ntr[23221:23221+7]                                                                                                                  
    Out[70]: array([   0,    0,    0, 7958, 7959, 7960, 7961], dtype=int32)

    In [73]: ntr[23221:23221+7]-1
    Out[73]: array([  -1,   -1,   -1, 7957, 7958, 7959, 7960], dtype=int32)

    In [72]: cf.tran[7957:7961]                         ## transforms on the CSG nodes of lv 118                                                                           
    Out[72]:                                            ## ARE THESE PRODUCTS ?   
    array([[[  1.361,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.361,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   ,   0.   ,   1.   ]],

           [[  1.   ,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   , -19.4  ,   1.   ]],

           [[  1.376,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.376,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   ,   0.   ,   1.   ]],

           [[  1.   ,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   , -19.9  ,   1.   ]]], dtype=float32)

    In [58]: snd.Label(3,8),st.f.csg.node[st.csg.lvid == 118],st.soname_[118]                                                                    
    Out[58]: 
    ('        ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf',
     array([[543,   2,   0, 545,   0,  -1, 544, 118, 103, 328, 328, 203,   0,   0,   0,   0],
            [544,   2,   1, 545,   0,  -1,  -1, 118, 105, 329, 329, 204,   0,   0,   0,   0],
            [545,   1,   0, 549,   2, 543, 548, 118,   1,  -1,  -1,  -1,   0,   0,   0,   0],
            [546,   2,   0, 548,   0,  -1, 547, 118, 103, 330, 330, 205,   0,   0,   0,   0],
            [547,   2,   1, 548,   0,  -1,  -1, 118, 105, 331, 331, 206,   0,   0,   0,   0],
            [548,   1,   1, 549,   2, 546,  -1, 118,   1,  -1,  -1, 207,   0,   0,   0,   0],
            [549,   0,  -1,  -1,   2, 545,  -1, 118,   3,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32),
     b'NNVTMCPPMTsMask')

    In [61]: st.f.csg.xform[203:208].reshape(-1,4,4)
    Out[61]: 
    array([[[  1.361,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.361,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   ,   0.   ,   1.   ]],

           [[  1.   ,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   , -19.4  ,   1.   ]],

           [[  1.376,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.376,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   ,   0.   ,   1.   ]],

           [[  1.   ,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   , -19.9  ,   1.   ]],

           [[  1.   ,   0.   ,   0.   ,   0.   ],
            [  0.   ,   1.   ,   0.   ,   0.   ],
            [  0.   ,   0.   ,   1.   ,   0.   ],
            [  0.   ,   0.   ,   0.   ,   1.   ]]])



    In [103]: LVID=128 ; st.soname_[LVID],snode.Label(6,11),stf.nds[stf.nds[:,snode.lv]==LVID]                                                   
    Out[103]: 
    (b'NNVTMCPPMTsMask_virtual',
     '           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      ro      bd',
     array([[ 70993,      6,   3065,  67846,      3,  70994,  71004,    128,      2,      2,    506,      2,      0,     26],
            [ 71018,      6,   3067,  67846,      3,  71019,  71029,    128,      4,      4,    508,      2,      1,     26],
            [ 71043,      6,   3069,  67846,      3,  71044,  71054,    128,      6,      6,    510,      2,      2,     26],
            [ 71250,      6,   3084,  67846,      3,  71251,  71261,    128,     21,     21,    525,      2,      3,     26],
            [ 71261,      6,   3085,  67846,      3,  71262,  71272,    128,     22,     22,    526,      2,      4,     26],
            [ 71272,      6,   3086,  67846,      3,  71273,  71283,    128,     23,     23,    527,      2,      5,     26],
            [ 71283,      6,   3087,  67846,      3,  71284,  71294,    128,     24,     24,    528,      2,      6,     26],

    In [104]: LVID=118 ; st.soname_[LVID],snode.Label(6,11),stf.nds[stf.nds[:,snode.lv]==LVID]                                                   
    Out[104]: 
    (b'NNVTMCPPMTsMask',
     '           ix      dp      sx      pt      nc      fc      sx      lv      cp      se      sx      ri      ro      bd',
     array([[ 70994,      7,      0,  70993,      0,     -1,  70995,    118,      0,     -1,     -1,      2,      0,     27],
            [ 71019,      7,      0,  71018,      0,     -1,  71020,    118,      0,     -1,     -1,      2,      1,     27],
            [ 71044,      7,      0,  71043,      0,     -1,  71045,    118,      0,     -1,     -1,      2,      2,     27],
            [ 71251,      7,      0,  71250,      0,     -1,  71252,    118,      0,     -1,     -1,      2,      3,     27],
            [ 71262,      7,      0,  71261,      0,     -1,  71263,    118,      0,     -1,     -1,      2,      4,     27],
            [ 71273,      7,      0,  71272,      0,     -1,  71274,    118,      0,     -1,     -1,      2,      5,     27],




combining the structural transforms with the csg transforms ?
----------------------------------------------------------------



GParts::applyPlacementTransform does::

    1243     for(unsigned i=0 ; i < ni ; i++)
    1244     {
    1245         nmat4triple* tvq = m_tran_buffer->getMat4TriplePtr(i) ;
    1247         bool match = true ;
    1248         const nmat4triple* ntvq = nmat4triple::make_transformed( tvq, placement, reversed, "GParts::applyPlacementTransform", match );
    1251         if(!match) mismatch.push_back(i);
    1253         m_tran_buffer->setMat4Triple( ntvq, i );
       

* placement is the structural transform


    266 const nmat4triple* nmat4triple::make_transformed(const nmat4triple* src, const glm::mat4& txf, bool reverse, const char* user, bool& match) // static
    267 {
    268     LOG(LEVEL) << "[ " << user ;
    269 
    270     nmat4triple perturb( txf );
    271     if(perturb.match == false)
    272     {
    273         LOG(error) << "perturb.match false : precision issue in inverse ? " ;
    274     }
    275 
    276     match = perturb.match ;
    277 
    278     std::vector<const nmat4triple*> triples ;
    279 
    280     if(reverse)
    281     {
    282         triples.push_back(src); 
    283         triples.push_back(&perturb);
    284     }
    285     else
    286     {
    287         triples.push_back(&perturb);
    288         triples.push_back(src); 
    289     }
    290 
    291     const nmat4triple* transformed = nmat4triple::product( triples, reverse );
    292 
    293     LOG(LEVEL) << "] " << user ;
    294     return transformed ;
    295 }




stree::get_itransform : where does the CSG transform inverse happen in old workflow
--------------------------------------------------------------------------------------

::

     323 void X4Solid::convertDisplacedSolid()
     324 {   
     325     const G4DisplacedSolid* const disp = static_cast<const G4DisplacedSolid*>(m_solid);
     326     G4VSolid* moved = disp->GetConstituentMovedSolid() ;
     327     assert( dynamic_cast<G4DisplacedSolid*>(moved) == NULL ); // only a single displacement is handled
     328     
     329     bool top = false ;  // never top of tree : expect to always be a boolean RHS
     330     X4Solid* xmoved = new X4Solid(moved, m_ok, top);
     331     setDisplaced(xmoved);
     332     
     333     nnode* a = xmoved->getRoot();
     334     
     335     LOG(LEVEL)
     336         << " a.csgname " << a->csgname()
     337         << " a.transform " << a->transform
     338         ;
     339     
     340     glm::mat4 xf_disp = X4Transform3D::GetDisplacementTransform(disp);
     341     
     342     bool update_global = false ;   // update happens later,  after tree completed
     343     a->set_transform( xf_disp, update_global );
     344     
     345     setRoot(a);
     346 }


     905 void nnode::set_transform( const glm::mat4& tmat, bool update_global )
     906 {
     907     const nmat4triple* add_transform = new nmat4triple(tmat) ;
     908 

     30 nmat4triple::nmat4triple(const glm::mat4& t_ )
     31     :
     32     match(true),
     33     t(t_),
     34     v(nglmext::invert_trs(t, match)),
     35     q(glm::transpose(v))
     36 {


     577 glm::mat4 nglmext::invert_trs( const glm::mat4& trs, bool& match )
     578 {
     579     bool verbose = false ;
     580     ndeco d ;
     581     polar_decomposition( trs, d, verbose) ;
     582     glm::mat4 isirit = d.isirit ;
     583     glm::mat4 i_trs = glm::inverse( trs ) ;
     584 
     585     NGLMCF cf(isirit, i_trs );
     586 
     587     if(!cf.match)
     588     {
     589         LOG(error) << "polar_decomposition inverse and straight inverse are mismatched " ;
     590         LOG(error) << cf.desc("ngmlext::invert_trs");
     591     }
     592 
     593     match = cf.match ;
     594 
     595     return isirit ;
     596 }



Transform references from the old GGeo created CSGNode
---------------------------------------------------------

TODO : recreate tran, itra using stree.h workflow

::


    In [17]: cf
    Out[17]: 
    /Users/blyth/.opticks/GEOM/J007/CSGFoundry
    min_stamp:2023-02-06 17:14:30.418383
    max_stamp:2023-02-06 17:14:32.968029
    age_stamp:6 days, 18:22:24.380705

             node :        (23547, 4, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/node.npy 

             itra :         (8179, 4, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/itra.npy 
             tran :         (8179, 4, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/tran.npy 

             prim :         (3259, 4, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/prim.npy 
         primname :              (3259,)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/primname.txt 

            solid :           (10, 3, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/solid.npy 
          mmlabel :                (10,)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/mmlabel.txt 

             inst :        (48477, 4, 4)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/inst.npy 

         meshname :               (152,)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/meshname.txt 
             meta :                 (8,)  : /Users/blyth/.opticks/GEOM/J007/CSGFoundry/meta.txt 


    In [5]: tr = cf.node.view(np.int32)[:,3,3] & 0x7fffffff

    In [18]: tr
    Out[18]: 
    array([   1,    2,    3,    0,    4,    5,    0,    6,    7,    8,    0,    9,   10,    0,   11,   12, ..., 8164, 8165, 8166, 8167, 8168, 8169, 8170, 8171, 8172, 8173, 8174, 8175, 8176, 8177, 8178,
           8179], dtype=int32)

    In [19]: tr[tr > 0]   ## looks monotonic when excluding the zero which mean no transform 
    Out[19]: 
    array([   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16, ..., 8164, 8165, 8166, 8167, 8168, 8169, 8170, 8171, 8172, 8173, 8174, 8175, 8176, 8177, 8178,
           8179], dtype=int32)

    In [20]: tr[tr > 0].shape
    Out[20]: (8179,)

    In [23]: np.all( tr[tr > 0] == np.arange(1,8180) )   ## confirmed, tr refs from the node are never duplicated 
    Out[23]: True

    In [8]: tr.min(), tr.max()
    Out[8]: (0, 8179)

    In [10]: u_tr, n_tr = np.unique(tr, return_counts=True)

    In [12]: np.all( u_tr == np.arange(len(u_tr)) )
    Out[12]: True

    In [13]: np.all( n_tr[1:] == 1 )
    Out[13]: True

    In [14]: n_tr[0]  ## nodes without an associated transform
    Out[14]: 15368

    In [15]: cf.node.shape
    Out[15]: (23547, 4, 4)

    In [16]: 15368 + 8179
    Out[16]: 23547


Pick an lvid and see its transforms.

* below manually interleaves outputs from stree_load_test.sh and CSGFoundryLoadTest.sh 
* now added stree loading to CSGFoundryLoadTest.py so can allways see both at once

::

    In [27]: plv = cf.prim.view(np.int32)[:,1,1]   

    In [29]: u_plv, n_plv = np.unique(plv, return_counts=True)

    In [31]: u_plv.min(), u_plv.max()
    Out[31]: (0, 149)

    In [34]: cf.meshname[102]
    Out[34]: 'solidXJanchor'

    In [1]: st.find_lvid_("solidXJanchor")
    Out[1]: array([102])

    In [2]: ln = st.find_lvid_nodes(102) ; ln
    Out[3]: 
    array([70853, 70854, 70855, 70856, 70857, 70858, 70859, 70860, 70861, 70862, 70863, 70864, 70865, 70866, 70867, 70868, 70869, 70870, 70871, 70872, 70873, 70874, 70875, 70876, 70877, 70878, 70879,
           70880, 70881, 70882, 70883, 70884, 70885, 70886, 70887, 70888, 70889, 70890, 70891, 70892, 70893, 70894, 70895, 70896, 70897, 70898, 70899, 70900, 70901, 70902, 70903, 70904, 70905, 70906,
           70907, 70908])

    In [4]: ln.shape
    Out[4]: (56,)



    In [46]: snd.Label(3,7),st.get_csg(102)    #  only 3 nodes : union of cylinder and cone : only one transform in the prim
    Out[46]: 
    ('       ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf',
     array([[456,   1,   0, 458,   0,  -1, 457, 102, 108, 279, 279,  -1,   0,   0,   0,   0],
            [457,   1,   1, 458,   0,  -1,  -1, 102, 105, 280, 280, 169,   0,   0,   0,   0],
            [458,   0,  -1,  -1,   2, 456,  -1, 102,   1,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32))


    In [67]: st.f.csg.xform[169].reshape(4,4)   ## HMM CAN I FIND THE TRAN WITH THIS COMBINED ? YES: DONE BELOW
    Out[67]: 
    array([[  1. ,   0. ,   0. ,   0. ],
           [  0. ,   1. ,   0. ,   0. ],
           [  0. ,   0. ,   1. ,   0. ],
           [  0. ,   0. , -11.5,   1. ]])

    In [71]: cf.tran[7327] - cf.tran[7326]     ## THERE IS ROTATION : SO THIS WONT WORK 
    Out[71]: 
    array([[ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [-0.051, -1.121, 11.445,  0.   ]], dtype=float32)


    In [73]: np.dot( st.f.csg.xform[169].reshape(4,4) , cf.tran[7326] )
    Out[73]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.974, -1739.136, 17755.355,     1.   ]])

    In [76]: np.dot( st.f.csg.xform[169].reshape(4,4), st.f.m2w[ln[0]] )  ## PRODUCT OF STRUCTURAL AND CSG TRANSFORMS
    Out[76]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.974, -1739.136, 17755.354,     1.   ]])

     CSG snd::Brief_ num_nodes 3
     0 : - ix:  456 dp:    1 sx:    0 pt:  458     nc:    0 fc:   -1 ns:  457 lv:  102     tc:  108 pa:  279 bb:  279 xf:   -1    co
     1 : - ix:  457 dp:    1 sx:    1 pt:  458     nc:    0 fc:   -1 ns:   -1 lv:  102     tc:  105 pa:  280 bb:  280 xf:  169    cy
     2 : - ix:  458 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  456 ns:   -1 lv:  102     tc:    1 pa:   -1 bb:   -1 xf:   -1    un

     tr dmat4x4((0.045146, 0.994203, 0.097583, 0.000000), 
                (0.998971, -0.045363, -0.000000, 0.000000), 
                (0.004427, 0.097482, -0.995227, 0.000000), 
               (-78.973684, -1739.135560, 17755.354429, 1.000000))





    In [74]: cf.tran[7327]
    Out[74]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.974, -1739.136, 17755.355,     1.   ]], dtype=float32)


    In [75]: cf.tran[7326]
    Out[75]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.923, -1738.015, 17743.91 ,     1.   ]], dtype=float32)


     tr dmat4x4((0.045146, 0.994203, 0.097583, 0.000000), (0.998971, -0.045363, -0.000000, 0.000000), (0.004427, 0.097482, -0.995227, 0.000000), (-78.922777, -1738.014511, 17743.909314, 1.000000))


    In [8]: st.f.m2w[ln[0]]
    Out[8]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,    -0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.923, -1738.015, 17743.909,     1.   ]])







    In [39]: w = np.where( plv == 102)[0]

    In [40]: w.shape
    Out[40]: (56,)

::

    In [48]: np.arange( 7326, 7438, 2 )
    Out[48]: 
    array([7326, 7328, 7330, 7332, 7334, 7336, 7338, 7340, 7342, 7344, 7346, 7348, 7350, 7352, 7354, 7356, 7358, 7360, 7362, 7364, 7366, 7368, 7370, 7372, 7374, 7376, 7378, 7380, 7382, 7384, 7386, 7388,
           7390, 7392, 7394, 7396, 7398, 7400, 7402, 7404, 7406, 7408, 7410, 7412, 7414, 7416, 7418, 7420, 7422, 7424, 7426, 7428, 7430, 7432, 7434, 7436])

    In [49]: np.all( cf.prim.view(np.int32)[w,0,2] == np.arange( 7326, 7438, 2 ) )
    Out[49]: True




    In [11]: plv = cf.prim.view(np.int32)[:,1,1]
    In [12]: w = np.where( plv == 102)[0]
    In [13]: pto = cf.prim.view(np.int32)[w,0,2] ; pto   ## transform offsets for all prim that are lv 102
    Out[14]: 
    array([7326, 7328, 7330, 7332, 7334, 7336, 7338, 7340, 7342, 7344, 7346, 7348, 7350, 7352, 7354, 7356, 7358, 7360, 7362, 7364, 7366, 7368, 7370, 7372, 7374, 7376, 7378, 7380, 7382, 7384, 7386, 7388,
           7390, 7392, 7394, 7396, 7398, 7400, 7402, 7404, 7406, 7408, 7410, 7412, 7414, 7416, 7418, 7420, 7422, 7424, 7426, 7428, 7430, 7432, 7434, 7436], dtype=int32)



    In [15]: cf.tran[pto[0]]
    Out[15]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.923, -1738.015, 17743.91 ,     1.   ]], dtype=float32)

    In [8]: st.f.m2w[ln[0]]
    Out[8]: 
    array([[    0.045,     0.994,     0.098,     0.   ],
           [    0.999,    -0.045,    -0.   ,     0.   ],
           [    0.004,     0.097,    -0.995,     0.   ],
           [  -78.923, -1738.015, 17743.909,     1.   ]])





    In [16]: cf.tran[pto[1]]
    Out[16]: 
    array([[    0.045,     0.991,     0.129,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.006,     0.129,    -0.992,     0.   ],
           [ -104.167, -2293.933, 17680.506,     1.   ]], dtype=float32)

    In [9]: st.f.m2w[ln[1]]
    Out[9]: 
    array([[    0.045,     0.991,     0.129,     0.   ],
           [    0.999,    -0.045,     0.   ,     0.   ],
           [    0.006,     0.129,    -0.992,     0.   ],
           [ -104.167, -2293.933, 17680.505,     1.   ]])




    In [17]: cf.tran[pto[-1]]
    Out[17]: 
    array([[    0.548,    -0.831,     0.098,     0.   ],
           [   -0.835,    -0.55 ,     0.   ,     0.   ],
           [    0.054,    -0.081,    -0.995,     0.   ],
           [ -957.729,  1452.473, 17743.91 ,     1.   ]], dtype=float32)

    In [9]: st.f.m2w[ln[-1]]
    Out[9]: 
    array([[    0.548,    -0.831,     0.098,     0.   ],
           [   -0.835,    -0.55 ,     0.   ,     0.   ],
           [    0.054,    -0.081,    -0.995,     0.   ],
           [ -957.729,  1452.473, 17743.909,     1.   ]])



::

    In [24]: np.c_[cf.tran[pto[:]], st.f.m2w[ln[:]]]
    Out[24]: 
    array([[[     0.045,      0.994,      0.098,      0.   ,      0.045,      0.994,      0.098,      0.   ],
            [     0.999,     -0.045,      0.   ,      0.   ,      0.999,     -0.045,      0.   ,      0.   ],
            [     0.004,      0.097,     -0.995,      0.   ,      0.004,      0.097,     -0.995,      0.   ],
            [   -78.923,  -1738.015,  17743.91 ,      1.   ,    -78.923,  -1738.015,  17743.909,      1.   ]],

           [[     0.045,      0.991,      0.129,      0.   ,      0.045,      0.991,      0.129,      0.   ],
            [     0.999,     -0.045,      0.   ,      0.   ,      0.999,     -0.045,      0.   ,      0.   ],
            [     0.006,      0.129,     -0.992,      0.   ,      0.006,      0.129,     -0.992,      0.   ],
            [  -104.167,  -2293.933,  17680.506,      1.   ,   -104.167,  -2293.933,  17680.505,      1.   ]],

        ...

           [[     0.546,     -0.828,      0.129,      0.   ,      0.546,     -0.828,      0.129,      0.   ],
            [    -0.835,     -0.55 ,      0.   ,      0.   ,     -0.835,     -0.55 ,      0.   ,      0.   ],
            [     0.071,     -0.108,     -0.992,      0.   ,      0.071,     -0.108,     -0.992,      0.   ],
            [ -1264.067,   1917.058,  17680.506,      1.   ,  -1264.067,   1917.058,  17680.505,      1.   ]],

           [[     0.548,     -0.831,      0.098,      0.   ,      0.548,     -0.831,      0.098,      0.   ],
            [    -0.835,     -0.55 ,      0.   ,      0.   ,     -0.835,     -0.55 ,      0.   ,      0.   ],
            [     0.054,     -0.081,     -0.995,      0.   ,      0.054,     -0.081,     -0.995,      0.   ],
            [  -957.729,   1452.473,  17743.91 ,      1.   ,   -957.729,   1452.473,  17743.909,      1.   ]]])

    In [25]:                                  





AB comparison using CSGFoundryAB.sh
--------------------------------------

::

    ## rebuild and install after changes as lots of headeronly functionality 

    sy      
    om 
    u4
    om 
    c
    om 


    u4t
    ./U4TreeCreateTest.sh   ## Create stree from gdml
    ct
    ./CSGImportTest.sh      ## import stree into CSGFoundary and save 

    ## TODO: combine the above two steps
    ct
    ./CSGFoundryAB.sh       ## compare A:old and B:new CSGFoundry 



Missing itra tran and inst in B::


  : A.SSim                                             :                 None : 4 days, 3:38:40.838511 
  : A.solid                                            :           (10, 3, 4) : 4 days, 3:39:36.056485 
  : A.prim                                             :         (3259, 4, 4) : 4 days, 3:39:36.057583 
  : A.node                                             :        (23547, 4, 4) : 4 days, 3:39:36.441330 

  : A.mmlabel                                          :                   10 : 4 days, 3:39:37.611860 
  : A.primname                                         :                 3259 : 4 days, 3:39:36.056862 
  : A.meshname                                         :                  152 : 4 days, 3:39:37.612941 
  : A.meta                                             :                    8 : 4 days, 3:39:37.612404 

  : A.itra                                             :         (8179, 4, 4) : 4 days, 3:39:37.613551 
  : A.tran                                             :         (8179, 4, 4) : 4 days, 3:39:35.423639 

  : A.inst                                             :        (48477, 4, 4) : 4 days, 3:39:37.973285 




Where to do balancing and positivization in new workflow ?
-------------------------------------------------------------

Old::

    X4PhysicalVolume::ConvertSolid
    X4PhysicalVolume::ConvertSolid_ 
    X4PhysicalVolume::ConvertSolid_FromRawNode
    NTreeProcess::init
    NTreePositive::init 


CSG transforms : stree/scsg f.csg.xform only 240 items vs CSGFoundry A.tran with 8179 ? 
-----------------------------------------------------------------------------------------

CSGFoundry has thousands of CSG level tran,itra::

  : A.itra                                             :         (8179, 4, 4) : 4 days, 3:39:37.613551 
  : A.tran                                             :         (8179, 4, 4) : 4 days, 3:39:35.423639 

scsg only has 240 xform (thats a repetition factor of 34)::

    In [12]: f.csg 
    CMDLINE:/Users/blyth/opticks/sysrap/tests/stree_load_test.py
    csg.base:/Users/blyth/.opticks/GEOM/J007/CSGFoundry/SSim/stree/csg

      : csg.node                                           :            (637, 16) : 4 days, 21:34:31.826544 
      : csg.aabb                                           :             (387, 6) : 4 days, 21:34:31.827683 
      : csg.xform                                          :            (240, 16) : 4 days, 21:34:31.825574 
      : csg.NPFold_index                                   :                    4 : 4 days, 21:34:31.828374 
      : csg.param                                          :             (387, 6) : 4 days, 21:34:31.826033 


* presumably some kind of repetition in CSGFoundry, but elaborate on that 
* tracing in CSGFoundry provides the explanation

  * because CSGFoundry::addTran gets called from CSGFoundry::addNode are getting 
    significant repetition of CSG level transforms due to node repetition eg from the globals 

  * POTENTIAL FOR ENHANCEMENT HERE : BUT SOME RELOCATING OF GLOBALS IS DONE SOMEWHERE, SO NON-TRIVIAL  

::

    1366 CSGNode* CSGFoundry::addNode(CSGNode nd, const std::vector<float4>* pl, const Tran<double>* tr  )
    1367 {
    ...
    1371     unsigned globalNodeIdx = node.size() ;
    ...
    1404     if(tr)
    1405     {
    1406         unsigned trIdx = 1u + addTran(tr);  // 1-based idx, 0 meaning None
    1407         nd.setTransform(trIdx);
    1408     }
    1409 
    1410     node.push_back(nd);
    1411     last_added_node = node.data() + globalNodeIdx ;
    1412     return last_added_node ;
    1413 }


HMM actually a lower level CSG_GGeo_Convert::convertNode is used doing much the same::

     674 CSGNode* CSG_GGeo_Convert::convertNode(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
     675 {
     ...
     677     unsigned partOffset = comp->getPartOffset(primIdx) ;
     678     unsigned partIdx = partOffset + partIdxRel ;
     ...
     691     const Tran<float>* tv = nullptr ; 
     692     unsigned gtran = comp->getGTransform(partIdx);  // 1-based index, 0 means None
     693     if( gtran > 0 )
     694     {
     695         glm::mat4 t = comp->getTran(gtran-1,0) ;
     696         glm::mat4 v = comp->getTran(gtran-1,1); 
     697         tv = new Tran<float>(t, v); 
     698     }
     699 
     700     unsigned tranIdx = tv ?  1 + foundry->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
     701 
     702     // HMM: this is not using the higher level 
     703     // CSGFoundry::addNode with transform pointer argumnent 
     704 


Need to do something similar in CSGImport::importNode 
BUT first need the gtransforms, snd/scsg only has local transforms so far. 

::

     740 /**
     741 nnode::global_transform
     742 ------------------------
     743 
     744 NB parent links are needed
     745 
     746 Is invoked by nnode::update_gtransforms_r from each primitive, 
     747 whence parent links are followed up the tree until reaching root
     748 which has no parent. Along the way transforms are collected
     749 into the tvq vector in reverse hierarchical order from 
     750 leaf back up to root
     751 
     752 If a placement transform is present on the root node, that 
     753 is also collected. 
     754 
     755 * NB these are the CSG nodes, not structure nodes
     756 
     757 **/
     759 const nmat4triple* nnode::global_transform(nnode* n)
     760 {
     761     std::vector<const nmat4triple*> tvq ;
     762     nnode* r = NULL ;
     763     while(n)
     764     {
     765         if(n->transform) tvq.push_back(n->transform);
     766         r = n ;            // keep hold of the last non-NULL 
     767         n = n->parent ;
     768     }
     769 
     770     if(r->placement) tvq.push_back(r->placement);
     771 
     772 
     773     bool reverse = true ;
     774     const nmat4triple* gtransform= tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ;
     775 
     776     if(gtransform == NULL )  // make sure all primitives have a gtransform 
     777     {
     778         gtransform = nmat4triple::make_identity() ;
     779     }
     780     return gtransform ;
     781 }


TODO: trace where the placement transforms come from 
---------------------------------------------------------

::

     567 void GMergedMesh::mergeVolume( const GVolume* volume, bool selected)
     568 {
     569     const GNode* node = static_cast<const GNode*>(volume);
     570     const GNode* base = getCurrentBase();
     571     unsigned ridx = volume->getRepeatIndex() ; 
     572 
     573     GVolume* volume_ = const_cast<GVolume*>(volume);
     574     GMatrixF* transform = base ? volume_->getRelativeTransform(base) : volume->getTransform() ;     // base relative OR global transform
     575 
     576     if( node == base ) assert( transform->isIdentity() );
     577     if( ridx == 0 )    assert( base == NULL && "expecting NULL base for ridx 0" );
     ...
     600     bool admit = selected ;
     601 
     602 
     603     if(admit)
     604     {
     605         mergeVolumeTransform(transform) ;        // "m_transforms[m_cur_volume]" 
     606         mergeVolumeBBox(vertices, num_vert);     // m_bbox[m_cur_volume], m_center_extent[m_cur_volume]  
     607         mergeVolumeIdentity(volume, selected );  // m_nodeinfo[m_cur_volume], m_identity[m_cur_volume], m_meshes[m_cur_volume]
     608 
     609         m_cur_volume += 1 ;    // irrespective of selection, as prefer absolute volume indexing 
     610         // NB admit: must parallel what is counted in countVolume 
     611     }
     613     if(selected)
     614     {
     615         mergeVolumeVertices( num_vert, vertices, normals );  // m_vertices, m_normals
     616 
     617         unsigned* node_indices     = volume->getNodeIndices();
     618         unsigned* boundary_indices = volume->getBoundaryIndices();
     619         unsigned* sensor_indices   = volume->getSensorIndices();
     620 
     621         mergeVolumeFaces( num_face, faces, node_indices, boundary_indices, sensor_indices  ); // m_faces, m_nodes, m_boundaries, m_sensors
     622 
     623         GPt* pt = volume->getPt();  // analytic 
     624         mergeVolumeAnalytic( pt, transform);     // relative or global transform as appropriate becoming the GPt placement, and collects into GMergedMesh::m_pts 
     625 
     626         // offsets with the flat arrays
     627         m_cur_vertices += num_vert ;
     628         m_cur_faces    += num_face ;
     629     }
     630 }




::

     916 /**
     917 ``GMergedMesh::mergeVolumeAnalytic``
     918 -------------------------------------
     919 
     920 Canonically invoked from ``GMergedMesh::mergeVolume``
     921 
     922 ``GPt`` instance from the volume are instanciated within ``X4PhysicalVolume::convertNode``.
     923 
     924 Only here does it become possible to set the placement transform into the GPt.
     925 This collects the GPt with its placement into GPts::m_pts, 
     926 which is then persisted into the geocache. 
     927 
     928 With repeated geometry one GPt instance for each GVolume is collected into GPts m_pts. 
     929 
     930 **/
     931 
     932 void GMergedMesh::mergeVolumeAnalytic( GPt* pt, GMatrixF* transform)
     933 {
     934     if(!pt) return ;
     935 
     936     const float* data = static_cast<float*>(transform->getPointer());
     937 
     938     glm::mat4 placement = glm::make_mat4( data ) ;
     939 
     940     pt->setPlacement(placement);
     941 
     942     m_pts->add( pt );
     943 }


Placement settings have to be late, after factorization. So NNode 
update_gtransform is way too soon::

     198 /**
     199 GParts::Create from GPts
     200 --------------------------
     201 
     202 Canonically invoked from ``GGeo::deferredCreateGParts``
     203 by ``GGeo::postLoadFromCache`` or ``GGeo::postDirectTranslation``.
     204 
     205 The (GPt)pt from each GVolume yields a per-volume (GParts)parts instance
     206 that is added to the (GParts)com instance.
     207 
     208 ``GParts::Create`` from ``GPts`` duplicates the standard precache GParts 
     209 in a deferred postcache manner using NCSG solids persisted with GMeshLib 
     210 and the requisite GParts arguments (spec, placement transforms) persisted by GPts 
     211 together with the GGeoLib merged meshes.  
     212 
     213 Note that GParts::applyPlacementTransform is applied to each individual 
     214 GParts object prior to combination into a composite GParts using the placement 
     215 transform collected into the GPt objects transported via GPts.
     216 
     217 GMergedMesh::mergeVolume
     218 GMergedMesh::mergeVolumeAnalytic
     219      combining and applying placement transform
     220 
     221 * GPts instances for each mergedMesh and merged from individual volume GPts. 
     222 
     223 * testing this with GPtsTest, using GParts::Compare 
     224 
     225 * notice how a combined GParts instance is contatenated from individual GParts instance for each GPt 
     226   using the referenced NCSG 
     227 
     228 * there is one combined GParts instance for each GMergedMesh which concatenates together the 
     229   analytic CSG buffers for all the "layers" of the GMergedMesh   
     230 
     231 **/
     ...
     245 GParts* GParts::Create(
     246     const Opticks* ok,
     247     const GPts* pts,
     248     const std::vector<const NCSG*>& solids,
     249     unsigned* num_mismatch_pt,
     250     std::vector<glm::mat4>* mismatch_placements,
     251     int imm ) // static
     252 {
     253     plog::Severity level = DEBUG == 0 ? LEVEL : info ;
     254     unsigned num_pt = pts->getNumPt();
     ...
     266     GParts* com = new GParts() ;
     267     com->setOpticks(ok);
     268     com->setRepeatIndex(imm);
     269 
     270     unsigned verbosity = 0 ;
     271     std::vector<unsigned> mismatch_pt ;
     272 
     273     for(unsigned i=0 ; i < num_pt ; i++)
     274     {
     275         const GPt* pt = pts->getPt(i); //  GPt holds structural tree transforms and indices collected in X4PhysicalVolume::convertNode 
     276         int   lvIdx = pt->lvIdx ;
     277         int   ndIdx = pt->ndIdx ;
     278         const std::string& spec = pt->getSpec() ;
     279         const glm::mat4& placement = pt->getPlacement() ;
     ...
     289         assert( lvIdx > -1 );
     290         const NCSG* csg = unsigned(lvIdx) < solids.size() ? solids[lvIdx] : NULL ;
     291         assert( csg );
     292 
     293         GParts* parts = GParts::Make( csg, spec.c_str(), ndIdx );
     294 
     295         unsigned num_mismatch = 0 ;
     296         parts->applyPlacementTransform( placement, verbosity, num_mismatch );   // this changes parts:m_tran_buffer
     297         if(num_mismatch > 0 ) RecordMismatch( mismatch_pt, mismatch_placements, placement, i, lvIdx, ndIdx, num_mismatch );
     298 
     299         parts->dumpTran("parts");
     300         com->add( parts );
     301     }
     302 
     303     com->dumpTran("com");
     304     DumpMismatch( num_mismatch_pt, mismatch_pt );
     305 
     306     LOG(level) << "]" ;
     307     return com ;
     308 }


::

    1185 /**
    1186 GParts::applyPlacementTransform
    1187 --------------------------------
    1188 
    1189 1. transforms the entire m_tran_buffer with the passed transform, 
    1190    to avoid leaving behind constituents this means that every constituent
    1191    must have an associated transform, **even if its the identity transform**
    1192 
    1193 * This was formerly invoked from GGeo::prepare...GMergedMesh::mergeVolumeAnalytic
    1194 * Now it is invoked by GParts::Create 
    1195 
    1196 **/
    1197 
    1198 void GParts::applyPlacementTransform(GMatrix<float>* gtransform, unsigned verbosity, unsigned& num_mismatch )
    1199 {
    1200     const float* data = static_cast<float*>(gtransform->getPointer());
    1201     if(verbosity > 2)
    1202     nmat4triple::dump(data, "GParts::applyPlacementTransform gtransform:" );
    1203     glm::mat4 placement = glm::make_mat4( data ) ;
    1204 
    1205     applyPlacementTransform( placement, verbosity, num_mismatch );
    1206 }
    1207 

    1241     std::vector<unsigned> mismatch ;
    1242 
    1243     for(unsigned i=0 ; i < ni ; i++)
    1244     {
    1245         nmat4triple* tvq = m_tran_buffer->getMat4TriplePtr(i) ;
    1246 
    1247         bool match = true ;
    1248         const nmat4triple* ntvq = nmat4triple::make_transformed( tvq, placement, reversed, "GParts::applyPlacementTransform", match );
    1249                               //  ^^^^^^^^^^^^^^^^^^^^^^^ SUSPECT DOUBLE NEGATIVE RE REVERSED  ^^^^^^^
    1250 
    1251         if(!match) mismatch.push_back(i);
    1252 
    1253         m_tran_buffer->setMat4Triple( ntvq, i );
    1254     }


::

    1758 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1759 {
    ....
    1807     GPt* pt = new GPt( lvIdx, ndIdx, csgIdx, boundaryName.c_str() )  ;
    1808 
    1809     //
    1810     // Q: where is the GPt placement transform set ?
    1811     // A: by GMergedMesh::mergeVolume/GMergedMesh::mergeVolumeAnalytic 
    1812     //    using a base relative or global transform depending on ridx
    1813     //
    1814     // WHY: because before analysis and resulting "factorization" 
    1815     //      of the geometry cannot know the appropriate placement transform to assign to he GPt
    1816     //       
    1817     // Local and global transform triples are collected below into GVolume with::
    1818     //       
    1819     //     GVolume::setLocalTransform(ltriple)
    1820     //     GVolume::setGlobalTransform(gtriple)
    1821     //
    1822     //  Those are the ingredients that later are used to get the appropriate 
    1823     //  combination of transforms.
    1824 
    1825 
    1826     glm::mat4 xf_local_t = X4Transform3D::GetObjectTransform(pv);
    1827 






::

     674 CSGNode* CSG_GGeo_Convert::convertNode(const GParts* comp, unsigned primIdx, unsigned partIdxRel )
     675 {
     676     unsigned repeatIdx = comp->getRepeatIndex();  // set in GGeo::deferredCreateGParts
     677     unsigned partOffset = comp->getPartOffset(primIdx) ;
     678     unsigned partIdx = partOffset + partIdxRel ;
     679     unsigned idx = comp->getIndex(partIdx);
     680     assert( idx == partIdx );
     681     unsigned boundary = comp->getBoundary(partIdx); // EXPT
     682 
     683     std::string tag = comp->getTag(partIdx);
     684     unsigned tc = comp->getTypeCode(partIdx);
     685     bool is_list = CSG::IsList((OpticksCSG_t)tc) ;
     686     int subNum = is_list ? comp->getSubNum(partIdx) : -1 ;
     687     int subOffset = is_list ? comp->getSubOffset(partIdx) : -1 ;
     688 
     689 
     690     // TODO: transform handling in double, narrowing to float at the last possible moment 
     691     const Tran<float>* tv = nullptr ;
     692     unsigned gtran = comp->getGTransform(partIdx);  // 1-based index, 0 means None
     693     if( gtran > 0 )
     694     {
     695         glm::mat4 t = comp->getTran(gtran-1,0) ;
     696         glm::mat4 v = comp->getTran(gtran-1,1);
     697         tv = new Tran<float>(t, v);
     698     }
     699 
     700     unsigned tranIdx = tv ?  1 + foundry->addTran(tv) : 0 ;   // 1-based index referencing foundry transforms
     701 




WIP : CSG transforms
-----------------------


More modern transform handling (for structure) in stree::get_m2w_product

* need something similar for CSG snd starting with get_ancestors following parent links 



* HMM is G4Ellipsoid scale Xform added ? YEP snd::SetNodeXForm(root, scale ); 


::

    In [15]: f.csg.node.shape
    Out[15]: (624, 16)

    In [12]: f.csg.node[:,snd.xf].min(), f.csg.node[:,snd.xf].max()   # the snd refs the xform 
    Out[12]: (-1, 239)

    In [9]: f.csg.xform.shape
    Out[9]: (240, 16)

    In [7]: np.unique( f.csg.node[:,snd.xf], return_counts=True )  # many -1 "null" but only one of 0 to 239
    Out[7]: 
    (array([ -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36, ...
            232, 233, 234, 235, 236, 237, 238, 239], dtype=int32),
     array([389,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
              1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
              1,   1,   1,   1,   1,   1,   1,   1]))






persisted stree.h looks to have lots of debugging extras not in CSGFoundry
------------------------------------------------------------------------------ 

* TODO: review U4Tree creation of the stree
* TODO: document stree.h arrays in constexpr notes
* TODO: make non-essentials optional in the persisted folder
* TODO: comparing transforms with CSGFoundry ones, work out how to CSGImport 

::

    f.base:/Users/blyth/.opticks/GEOM/J007/CSGFoundry/SSim/stree

      : f.inst                                             :        (48477, 4, 4) : 4 days, 19:50:27.711199 
      : f.iinst                                            :        (48477, 4, 4) : 4 days, 19:51:11.766482 
      : f.inst_f4                                          :        (48477, 4, 4) : 4 days, 19:50:15.080793 
      : f.iinst_f4                                         :        (48477, 4, 4) : 4 days, 19:51:03.021160 
      : f.inst_nidx                                        :             (48477,) : 4 days, 19:50:14.668612 

      : f.nds                                              :         (422092, 14) : 4 days, 19:49:13.288836 
      : f.gtd                                              :       (422092, 4, 4) : 4 days, 19:51:24.749130 
      : f.m2w                                              :       (422092, 4, 4) : 4 days, 19:49:56.715706 
      : f.w2m                                              :       (422092, 4, 4) : 4 days, 19:48:49.592172 
      : f.subs                                             :               422092 : 4 days, 19:49:10.533278 
      : f.digs                                             :               422092 : 4 days, 19:51:53.722598 

      : f.rem                                              :           (3089, 14) : 4 days, 19:49:12.776762 
      : f.factor                                           :              (9, 12) : 4 days, 19:51:53.722111 

      : f.sensor_id                                        :             (46116,) : 4 days, 19:49:11.631723 

      : f.csg                                              :                 None : 4 days, 19:48:49.587808 
      : f.soname                                           :                  150 : 4 days, 19:49:11.630989 

      : f.bd                                               :              (54, 4) : 4 days, 19:52:00.140095 
      : f.bd_names                                         :                   54 : 4 days, 19:52:00.139822 

      : f.surface                                          :                 None : 4 days, 19:48:49.258335 
      : f.suname                                           :                   46 : 4 days, 19:49:10.532493 
      : f.suindex                                          :                (46,) : 4 days, 19:49:10.532860 

      : f.material                                         :                 None : 4 days, 19:48:49.587773 
      : f.mtline                                           :                (20,) : 4 days, 19:49:56.714553 
      : f.mtname                                           :                   20 : 4 days, 19:49:56.714169 
      : f.mtindex                                          :                (20,) : 4 days, 19:49:56.714992 



Back to ct:CSGFoundryAB.sh 
---------------------------

Extra meshname in A from the unbalanced alt (names appear twice)::

    In [15]: A.meshname[149:]
    Out[15]: ['sWorld', 'solidSJReceiverFastern', 'uni1']

    In [16]: B.meshname[149:]
    Out[16]: ['sWorld0x59dfbe0']


* DONE: trim the 0x ref in B 


Comparing CSGFoundry
----------------------

* lvid 93, 99 are very different : A balanced, B not  
* A also positivized, B not 

::

    In [12]: A.base
    Out[12]: '/Users/blyth/.opticks/GEOM/J007/CSGFoundry'

    In [13]: B.base
    Out[13]: '/tmp/blyth/opticks/CSGImportTest/CSGFoundry'

    In [15]: A.prim.shape, B.prim.shape
    Out[15]: ((3259, 4, 4), (3259, 4, 4))

    In [16]: A.node.shape, B.node.shape
    Out[16]: ((23547, 4, 4), (25435, 4, 4))

    In [11]: np.c_[A.prim.view(np.int32)[:,0,:2],B.prim.view(np.int32)[:,0,:2]][:2200]
    Out[11]: 
    array([[    1,     0,     1,     0],
           [    1,     1,     1,     1],
           [    1,     2,     1,     2],
           [    3,     3,     3,     3],
           [    3,     6,     3,     6],
           ...,
           [    7, 14149,     7, 14149],
           [    7, 14156,     7, 14156],
           [    7, 14163,     7, 14163],
           [    7, 14170,     7, 14170],
           [    7, 14177,     7, 14177]], dtype=int32)


    In [18]: np.c_[A.prim.view(np.int32)[:,1],B.prim.view(np.int32)[:,1]][:1000]
    Out[18]: 
    array([[  0, 149,   0,   0,   0,  -1,   0,   0],
           [  1,  17,   0,   1,   1,  -1,   0,   0],
           [  2,   2,   0,   2,   2,  -1,   0,   0],
           [  3,   1,   0,   3,   3,  -1,   0,   0],
           [  4,   0,   0,   4,   4,  -1,   0,   0],
           ...,
           [995,  45,   0, 995, 995,  -1,   0,   0],
           [996,  45,   0, 996, 996,  -1,   0,   0],
           [997,  45,   0, 997, 997,  -1,   0,   0],
           [998,  45,   0, 998, 998,  -1,   0,   0],
           [999,  45,   0, 999, 999,  -1,   0,   0]], dtype=int32)


9 prim have mismatched numNode (8 are contiguous primIdx from ridx 0)::

    In [12]: mi = np.where( A.prim.view(np.int32)[:,0,0]  != B.prim.view(np.int32)[:,0,0] ) ; mi 
    Out[13]: (array([2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 3126]),)

    In [22]: np.c_[A.prim.view(np.int32)[mi,0],B.prim.view(np.int32)[mi,0]]
    Out[22]: 
    array([[[   15, 15209,  6672,     0,   127, 15209,     0,     0],
            [   15, 15224,  6680,     0,   127, 15336,     0,     0],
            [   15, 15239,  6688,     0,   127, 15463,     0,     0],
            [   15, 15254,  6696,     0,   127, 15590,     0,     0],
            [   15, 15269,  6704,     0,   127, 15717,     0,     0],
            [   15, 15284,  6712,     0,   127, 15844,     0,     0],
            [   15, 15299,  6720,     0,   127, 15971,     0,     0],
            [   15, 15314,  6728,     0,   127, 16098,     0,     0],
            [   31, 23372,  8032,     0,  1023, 24268,     0,     0]]], dtype=int32)

    ## numNode much bigger for B  (is A using CSG_CONTIGUOUS?)
    ## Looks like A is balanced but B is not. 
    ##
    ## B misses the tranOffset                                                          


    In [18]: A.meshname[93]
    Out[18]: 'solidSJReceiverFastern'

    In [19]: B.meshname[93]
    Out[19]: 'solidSJReceiverFastern0x5bc98c0'

    In [21]: A.meshname[99]
    Out[21]: 'uni1'

    In [20]: B.meshname[99]
    Out[20]: 'uni10x5a93440'


    In [17]: np.c_[A.prim.view(np.int32)[mi,1],B.prim.view(np.int32)[mi,1]]
    Out[17]: 
    array([[[2375,   93,    0, 2375, 2375,   93,    0, 2375],
            [2376,   93,    0, 2376, 2376,   93,    0, 2376],
            [2377,   93,    0, 2377, 2377,   93,    0, 2377],
            [2378,   93,    0, 2378, 2378,   93,    0, 2378],
            [2379,   93,    0, 2379, 2379,   93,    0, 2379],
            [2380,   93,    0, 2380, 2380,   93,    0, 2380],
            [2381,   93,    0, 2381, 2381,   93,    0, 2381],
            [2382,   93,    0, 2382, 2382,   93,    0, 2382],
            [   0,   99,    6,    0,    0,   99,    6,    0]]], dtype=int32)

    ## matched: sbtIndexOffset, meshIdx, repeatIdx, primIdx  


    In [25]: A.node[15209:15209+15].view(np.int32)[:,3,2:]
    Out[25]: 
    array([[          1,           0],
           [          1,           0],
           [          1,           0],
           [          1,           0],
           [          1,           0],
           [          2,           0],
           [          1,           0],
           [        110,        6673],
           [        110,        6674],
           [        110,        6675],
           [        110,        6676],
           [        105,        6677],
           [        105, -2147476970],
           [        110,        6679],
           [        110,        6680]], dtype=int32)

           110:box3 105:cyl 1:uni 2:intersect


    In [26]: B.node[15209:15209+127].view(np.int32)[:,3,2:]
    Out[26]: 
    array([[  1,   0],
           [  1,   0],
           [  1,   0],
           [  1,   0],
           [110,   0],
           [110,   0],
           [110,   0],
           [  1,   0],
           [110,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  1,   0],
           [110,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           [  0,   0],
           ...




Old workflow refs
-------------------

NCSG::export
    writes nodetree into transport buffers 

NCSG::export_tree
NCSG::export_list
NCSG::export_leaf

NCSG::export_tree_list_prepare
    explains subNum/subOffet in serialization 
    of trees with list nodes

nnode::find_list_nodes_r
nnode::is_list
    CSG::IsList(type)   CSG_CONTIGUOUS or CSG_DISCONTIGUOUS or CSG_OVERLAP      

nnode::subNum
nnode::subOffset

    CSG::IsCompound

CSGNode re:subNum subOffset
    Used by compound node types such as CSG_CONTIGUOUS, CSG_DISCONTIGUOUS and 
    the rootnode of boolean trees CSG_UNION/CSG_INTERSECTION/CSG_DIFFERENCE...
    Note that because subNum uses q0.u.x and subOffset used q0.u.y 
    this should not be used for leaf nodes. 

NCSG::export_tree_r
    assumes pure binary tree serializing to 2*idx+1 2*idx+2 




Consider lvid:103
---------------------

::

    CSGImport::importPrim@246:  primIdx 3078 lvid 103 num_nd  17 num_non_binary   0 max_binary_depth   6 : solidXJfixture0x5bbd6b0
    snd::render_v - ix:  475 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  469 ns:   -1 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 11
    snd::render_v - ix:  469 dp:    1 sx:    0 pt:  475     nc:    2 fc:  467 ns:  474 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 9
    snd::render_v - ix:  467 dp:    2 sx:    0 pt:  469     nc:    2 fc:  465 ns:  468 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 7
    snd::render_v - ix:  465 dp:    3 sx:    0 pt:  467     nc:    2 fc:  463 ns:  466 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 5
    snd::render_v - ix:  463 dp:    4 sx:    0 pt:  465     nc:    2 fc:  461 ns:  464 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 3
    snd::render_v - ix:  461 dp:    5 sx:    0 pt:  463     nc:    2 fc:  459 ns:  462 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:   -1    di ordinal 1
    snd::render_v - ix:  459 dp:    6 sx:    0 pt:  461     nc:    0 fc:   -1 ns:  460 lv:  103     tc:  105 pa:  281 bb:  281 xf:   -1    cy ordinal 0
    snd::render_v - ix:  460 dp:    6 sx:    1 pt:  461     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  105 pa:  282 bb:  282 xf:   -1    cy ordinal 2
    snd::render_v - ix:  462 dp:    5 sx:    1 pt:  463     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  283 bb:  283 xf:  170    bo ordinal 4
    snd::render_v - ix:  464 dp:    4 sx:    1 pt:  465     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  284 bb:  284 xf:  171    bo ordinal 6
    snd::render_v - ix:  466 dp:    3 sx:    1 pt:  467     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  285 bb:  285 xf:  172    bo ordinal 8
    snd::render_v - ix:  468 dp:    2 sx:    1 pt:  469     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  286 bb:  286 xf:  173    bo ordinal 10
    snd::render_v - ix:  474 dp:    1 sx:    1 pt:  475     nc:    2 fc:  472 ns:   -1 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:  176    di ordinal 15
    snd::render_v - ix:  472 dp:    2 sx:    0 pt:  474     nc:    2 fc:  470 ns:  473 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un ordinal 13
    snd::render_v - ix:  470 dp:    3 sx:    0 pt:  472     nc:    0 fc:   -1 ns:  471 lv:  103     tc:  110 pa:  287 bb:  287 xf:   -1    bo ordinal 12
    snd::render_v - ix:  471 dp:    3 sx:    1 pt:  472     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  288 bb:  288 xf:  174    bo ordinal 14
    snd::render_v - ix:  473 dp:    2 sx:    1 pt:  474     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  289 bb:  289 xf:  175    bo ordinal 16
    *CSGImport::importPrim@256: 
    snd::rbrief
    - ix:  475 dp:    0 sx:   -1 pt:   -1     nc:    2 fc:  469 ns:   -1 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  469 dp:    1 sx:    0 pt:  475     nc:    2 fc:  467 ns:  474 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  467 dp:    2 sx:    0 pt:  469     nc:    2 fc:  465 ns:  468 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  465 dp:    3 sx:    0 pt:  467     nc:    2 fc:  463 ns:  466 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  463 dp:    4 sx:    0 pt:  465     nc:    2 fc:  461 ns:  464 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  461 dp:    5 sx:    0 pt:  463     nc:    2 fc:  459 ns:  462 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:   -1    di
    - ix:  459 dp:    6 sx:    0 pt:  461     nc:    0 fc:   -1 ns:  460 lv:  103     tc:  105 pa:  281 bb:  281 xf:   -1    cy
    - ix:  460 dp:    6 sx:    1 pt:  461     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  105 pa:  282 bb:  282 xf:   -1    cy
    - ix:  462 dp:    5 sx:    1 pt:  463     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  283 bb:  283 xf:  170    bo
    - ix:  464 dp:    4 sx:    1 pt:  465     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  284 bb:  284 xf:  171    bo
    - ix:  466 dp:    3 sx:    1 pt:  467     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  285 bb:  285 xf:  172    bo
    - ix:  468 dp:    2 sx:    1 pt:  469     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  286 bb:  286 xf:  173    bo
    - ix:  474 dp:    1 sx:    1 pt:  475     nc:    2 fc:  472 ns:   -1 lv:  103     tc:    3 pa:   -1 bb:   -1 xf:  176    di
    - ix:  472 dp:    2 sx:    0 pt:  474     nc:    2 fc:  470 ns:  473 lv:  103     tc:    1 pa:   -1 bb:   -1 xf:   -1    un
    - ix:  470 dp:    3 sx:    0 pt:  472     nc:    0 fc:   -1 ns:  471 lv:  103     tc:  110 pa:  287 bb:  287 xf:   -1    bo
    - ix:  471 dp:    3 sx:    1 pt:  472     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  288 bb:  288 xf:  174    bo
    - ix:  473 dp:    2 sx:    1 pt:  474     nc:    0 fc:   -1 ns:   -1 lv:  103     tc:  110 pa:  289 bb:  289 xf:  175    bo


    snd::render width 17 height 6 mode 3

                                                un                          
                                                                            
                                        un                      di          
                                                                            
                                un          bo          un          bo      
                                                                            
                        un          bo              bo      bo              
                                                                            
                un          bo                                              
                                                                            
        di          bo                                                      
                                                                            
    cy      cy                                                              
                                                                            
                                                                            


    CSGImport::importNode_v@310:  idx 0
    CSGImport::importNode_v@310:  idx 1
    CSGImport::importNode_v@310:  idx 3
    CSGImport::importNode_v@310:  idx 7
    CSGImport::importNode_v@310:  idx 15
    CSGImport::importNode_v@310:  idx 31
    CSGImport::importNode_v@310:  idx 63
    CSGImport::importNode_v@310:  idx 64
    CSGImport::importNode_v@310:  idx 32
    CSGImport::importNode_v@310:  idx 16
    CSGImport::importNode_v@310:  idx 8
    CSGImport::importNode_v@310:  idx 4
    CSGImport::importNode_v@310:  idx 2
    CSGImport::importNode_v@310:  idx 5
    CSGImport::importNode_v@310:  idx 11
    CSGImport::importNode_v@310:  idx 12
    CSGImport::importNode_v@310:  idx 6

::

    In [7]: w = cf.prim.view(np.int32)[:,1,1] == 103

    In [10]: cf.prim[w].shape
    Out[10]: (56, 4, 4)

    In [13]: cf.prim[w].view(np.int32)[:,0]
    Out[13]: 
    array([[  127, 16087,  7438,     0],    ## numNode, nodeOffset, tranOffset, planOffset
           [  127, 16214,  7447,     0],
           [  127, 16341,  7456,     0],
           [  127, 16468,  7465,     0],
           [  127, 16595,  7474,     0],
           [  127, 16722,  7483,     0],
           [  127, 16849,  7492,     0],


    In [27]: np.c_[np.arange(127),cf.node[16087:16087+127,3,2:].view(np.int32) ]
    Out[27]: 
    array([[          0,           1,           0],      # i, tc, complement~gtransformIdx
           [          1,           1,           0],
           [          2,           2,           0],
           [          3,           1,           0],
           [          4,         110,        7439],
           [          5,           1,           0],
           [          6,         110, -2147476208],
           [          7,           1,           0],
           [          8,         110,        7441],
           [          9,           0,           0],
           [         10,           0,           0],
           [         11,         110,        7442],
           [         12,         110,        7443],
           [         13,           0,           0],
           [         14,           0,           0],
           [         15,           1,           0],
           [         16,         110,        7444],
           [         17,           0,           0],
           [         18,           0,           0],
           [         19,           0,           0],
           [         20,           0,           0],
           [         21,           0,           0],
           [         22,           0,           0],
           [         23,           0,           0],

           ...

           [         28,           0,           0],
           [         29,           0,           0],
           [         30,           0,           0],
           [         31,           2,           0],
           [         32,         110,        7445],
           [         33,           0,           0],
           [         34,           0,           0],
           [         35,           0,           0],

           ...

           [         61,           0,           0],
           [         62,           0,           0],
           [         63,         105,        7446],
           [         64,         105, -2147476201],
           [         65,           0,           0],
           [         66,           0,           0],



Consider lvid:100 base_steel which is a single polycone prim within ridx 7
-------------------------------------------------------------------------------

::

    CSGImport::importPrim@201:  primIdx    0 lvid 100 snd::GetLVID   7 : base_steel0x5b335a0





Hmm this stree still using contiguous::

    GEOM=J007 RIDX=7 ./sysrap/tests/stree_load_test.sh ana


     lv:100 nlv: 1                                         base_steel csg  7 tcn 105:cylinder 105:cylinder 11:contiguous 105:cylinder 105:cylinder 11:contiguous 3:difference 
    desc_csg lvid:100 st.f.soname[100]:base_steel 
            ix   dp   sx   pt   nc   fc   sx   lv   tc   pm   bb   xf
    array([[444,   2,   0, 446,   0,  -1, 445, 100, 105, 272, 272,  -1,   0,   0,   0,   0],
           [445,   2,   1, 446,   0,  -1,  -1, 100, 105, 273, 273,  -1,   0,   0,   0,   0],
           [446,   1,   0, 450,   2, 444, 449, 100,  11,  -1,  -1,  -1,   0,   0,   0,   0],
           [447,   2,   0, 449,   0,  -1, 448, 100, 105, 274, 274,  -1,   0,   0,   0,   0],
           [448,   2,   1, 449,   0,  -1,  -1, 100, 105, 275, 275,  -1,   0,   0,   0,   0],
           [449,   1,   1, 450,   2, 447,  -1, 100,  11,  -1,  -1,  -1,   0,   0,   0,   0],
           [450,   0,  -1,  -1,   2, 446,  -1, 100,   3,  -1,  -1,  -1,   0,   0,   0,   0]], dtype=int32)

    stree.descSolids numSolid:10 detail:0 





    CSGFoundry.descSolid ridx  7 label               r7 numPrim      1 primOffset   3127 lv_one 1 
     pidx 3127 lv 100 pxl    0 :                                         base_steel : no 23403 nn    7 tcn 2:intersection 1:union 2:intersection 105:cylinder 105:cylinder 105:!cylinder 105:!cylinder  






Further thoughts on CSGImport::importTree
----------------------------------------------

Further thoughts now solidifying into CSG/CSGImport.cc CSGImport::importTree

CSGSolid
    main role is to hold (numPrim, primOffset) : ie specify a contiguous range of CSGPrim
CSGPrim
    main role is to hold (numNode, nodeOffset) : ie specify a contiguous range of CSGNode 


Difficulty 1 : polycone compounds
------------------------------------

X4Solid::convertPolycone uses NTreeBuilder<nnode> to 
generate a suitably sized complete binary tree of CSG_ZERO gaps
and then populates it with the available nodes.

::

    1706 void X4Solid::convertPolycone()
    1707 {
    ....
    1785     std::vector<nnode*> outer_prims ;
    1786     Polycone_MakePrims( zp, outer_prims, m_name, true  );
    1787     bool dump = false ;
    1788     nnode* outer = NTreeBuilder<nnode>::UnionTree(outer_prims, dump) ;
    1789 

Whilst validating the conversion (because want to do identicality check between old and new workflows) 
will need to implement the same within snd/scsg for example steered from U4Solid::init_Polycone U4Polycone::Convert

Because snd uses n-ary tree can subsequently enhance to using CSG_CONTIGUOUS 
bringing the compound thru to the GPU. 




Thoughts : How difficulty to go direct Geant4 -> CSGFoundry ?
--------------------------------------------------------------

* Materials and surfaces : pretty easily as GGeo/GMaterialLib/GSurfaceLib 
  are fairly simple containers that can easily be replaced with more modern 
  and generic approaches using NPFold/NP/NPX/SSim

  * WIP: U4Material.h .cc U4Surface.h 
  * TODO: boundary array standardizing the data already collected by U4Material, U4Surface


* Structure : U4Tree/stree : already covers most of whats needed (all the
  transforms and doing the factorization)

* Solids : MOST WORK NEEDED : MADE RECENT PROGRESS WITH U4Solid

  * WIP: U4Solid snd scsg stree CSGFoundry::importTree
  * DECIDE NO NEED FOR C4 PKG  

  * intricate web of translation and primitives code across x4/npy/GGeo 
  * HOW TO PROCEED : START AFRESH : CONVERTING G4VSolid trees into CSGPrim/CSGNode trees

    * aiming for much less code : avoiding intermediaries

    * former persisting approach nnode/GParts/GPts needs to be ripped out
  
      * "ripping out" is the wrong approach : simpler to start without heed to 
        what was done before : other than where the code needed is directly 
        analogous : in which case methods should be extracted and name changed 

    * CSGFoundary/CSGSolid/CSGPrim/CSGNode : handles all the persisting much more simply 
      so just think of mapping CSG trees of G4VSolid into CSGPrim/CSGNode trees

    * U4SolidTree (developed for Z cutting) has lots of of general stuff 
      that could be pulled out into a U4Solid.h to handle the conversion 


   
Solids : Central Issue : How to handle the CSG node tree ?  
-------------------------------------------------------------

* Geant4 CSG trees have G4DisplacedSolid complications with transforms held in illogical places  
* can an intermediate node tree be avoided ? 
* old way far too complicated :  nnode, nsphere,..., NCSG, GParts, GPts, GMesh, ... 

  * nnode, nsphere,... : raw node tree
  * NCSG/GParts/GPts : persist related  
  * GMesh : triangles and holder of analytic GParts 


* U4SolidTree avoids an intermediate tree but at the expense of 
  having lots of maps keyed on the G4VSolid nodes of the tree 

  * it might actually be simpler with a transient minimal intermediate node tree 
    to provide a convenient place for annotation during conversion 


Solid Conversion Complications
---------------------------------

* balancing (this has been shown to cause missed intersects in some complex trees, so need to live without it)
* nudging : avoiding coincident faces 


Old Solid Conversion Code
---------------------------

::

    0890 void X4PhysicalVolume::convertSolids()
     891 {
     895     const G4VPhysicalVolume* pv = m_top ;
     896     int depth = 0 ;
     897     convertSolids_r(pv, depth);
     907 }

    0909 /**
     910 X4PhysicalVolume::convertSolids_r
     911 ------------------------------------
     912 
     913 G4VSolid is converted to GMesh with associated analytic NCSG 
     914 and added to GGeo/GMeshLib.
     915 
     916 If the conversion from G4VSolid to GMesh/NCSG/nnode required
     917 balancing of the nnode then the conversion is repeated 
     918 without the balancing and an alt reference is to the alternative 
     919 GMesh/NCSG/nnode is kept in the primary GMesh. 
     920 
     921 Note that only the nnode is different due to the balancing, however
     922 its simpler to keep a one-to-one relationship between these three instances
     923 for persistency convenience.
     924 
     925 Note that convertSolid is called for newly encountered lv
     926 in the postorder tail after the recursive call in order for soIdx/lvIdx
     927 to match Geant4. 
     928 
     929 **/
     930 
     931 void X4PhysicalVolume::convertSolids_r(const G4VPhysicalVolume* const pv, int depth)
     932 {
     933     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     934 
     935     // G4LogicalVolume::GetNoDaughters returns 1042:G4int, 1062:size_t
     936     for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ )
     937     {
     938         const G4VPhysicalVolume* const daughter_pv = lv->GetDaughter(i);
     939         convertSolids_r( daughter_pv , depth + 1 );
     940     }
     941 
     942     // for newly encountered lv record the tail/postorder idx for the lv
     943     if(std::find(m_lvlist.begin(), m_lvlist.end(), lv) == m_lvlist.end())
     944     {
     945         convertSolid( lv );
     946     } 
     947 }

    0961 void X4PhysicalVolume::convertSolid( const G4LogicalVolume* lv )
     962 {
     963     const G4VSolid* const solid = lv->GetSolid();
     964 
     965     G4String  lvname_ = lv->GetName() ;      // returns by reference, but take a copied value 
     966     G4String  soname_ = solid->GetName() ;   // returns by value, not reference
     967 
     968     const char* lvname = strdup(lvname_.c_str());  // may need these names beyond this scope, so strdup     
     969     const char* soname = strdup(soname_.c_str());
     ...
     986     GMesh* mesh = ConvertSolid(m_ok, lvIdx, soIdx, solid, soname, lvname );
     987     mesh->setX4SkipSolid(x4skipsolid);
     988 
    1001     m_ggeo->add( mesh ) ;
    1002 
    1003     LOG(LEVEL) << "] " << std::setw(4) << lvIdx ;
    1004 }   


    1104 GMesh* X4PhysicalVolume::ConvertSolid_( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* soname, const char* lvname,      bool balance_deep_tree ) // static
    1105 {   
    1129     const char* boundary = nullptr ; 
    1130     nnode* raw = X4Solid::Convert(solid, ok, boundary, lvIdx )  ;
    1131     raw->set_nudgeskip( is_x4nudgeskip );  
    1132     raw->set_pointskip( is_x4pointskip );
    1133     raw->set_treeidx( lvIdx );
    1134     
    1139     bool g4codegen = ok->isG4CodeGen() ;
    1140     
    1141     if(g4codegen) GenerateTestG4Code(ok, lvIdx, solid, raw);
    1142     
    1143     GMesh* mesh = ConvertSolid_FromRawNode( ok, lvIdx, soIdx, solid, soname, lvname, balance_deep_tree, raw );
    1144 
    1145     return mesh ;


::

    1156 GMesh* X4PhysicalVolume::ConvertSolid_FromRawNode( const Opticks* ok, int lvIdx, int soIdx, const G4VSolid* const solid, const char* soname, const ch     ar* lvname, bool balance_deep_tree,
    1157      nnode* raw)
    1158 {
    1159     bool is_x4balanceskip = ok->isX4BalanceSkip(lvIdx) ;
    1160     bool is_x4polyskip = ok->isX4PolySkip(lvIdx);   // --x4polyskip 211,232
    1161     bool is_x4nudgeskip = ok->isX4NudgeSkip(lvIdx) ;
    1162     bool is_x4pointskip = ok->isX4PointSkip(lvIdx) ;
    1163     bool do_balance = balance_deep_tree && !is_x4balanceskip ;
    1164 
    1165     nnode* root = do_balance ? NTreeProcess<nnode>::Process(raw, soIdx, lvIdx) : raw ;
    1166 
    1167     LOG(LEVEL) << " after NTreeProcess:::Process " ;
    1168 
    1169     root->other = raw ;
    1170     root->set_nudgeskip( is_x4nudgeskip );
    1171     root->set_pointskip( is_x4pointskip );
    1172     root->set_treeidx( lvIdx );
    1173 
    1174     const NSceneConfig* config = NULL ;
    1175 
    1176     LOG(LEVEL) << "[ before NCSG::Adopt " ;
    1177     NCSG* csg = NCSG::Adopt( root, config, soIdx, lvIdx );   // Adopt exports nnode tree to m_nodes buffer in NCSG instance
    1178     LOG(LEVEL) << "] after NCSG::Adopt " ;
    1179     assert( csg ) ;
    1180     assert( csg->isUsedGlobally() );
    1181 
    1182     bool is_balanced = root != raw ;
    1183     if(is_balanced) assert( balance_deep_tree == true );
    1184 
    1185     csg->set_balanced(is_balanced) ;
    1186     csg->set_soname( soname ) ;
    1187     csg->set_lvname( lvname ) ;
    1188 
    1189     LOG_IF(fatal, is_x4polyskip ) << " is_x4polyskip " << " soIdx " << soIdx  << " lvIdx " << lvIdx ;
    1190 
    1191     GMesh* mesh = nullptr ;
    1192     if(solid)
    1193     {
    1194         mesh =  is_x4polyskip ? X4Mesh::Placeholder(solid ) : X4Mesh::Convert(solid, lvIdx) ;
    1195     }
    1196     else
    1197     {





Old High Level Geometry Code
--------------------------------


::

    223 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    224 {   
    225     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    226     assert(world);
    227     wd = world ;
    228     
    229     //sim = SSim::Create();  // its created in ctor  
    230     assert(sim) ;
    231     
    232     stree* st = sim->get_tree(); 
    233     // TODO: sim argument, not st : or do SSim::Create inside U4Tree::Create 
    234     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    235 
    236     
    237     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    238     Opticks::Configure("--gparts_transform_offset --allownokey" );
    239     
    240     GGeo* gg_ = X4Geo::Translate(wd) ;
    241     setGeometry(gg_);
    242 }
    243 
    244 
    245 void G4CXOpticks::setGeometry(GGeo* gg_)
    246 {
    247     LOG(LEVEL);
    248     gg = gg_ ;
    249 
    250 
    251     CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ;
    252     setGeometry(fd_);
    253 }


::

     19 GGeo* X4Geo::Translate(const G4VPhysicalVolume* top)  // static 
     20 {
     21     bool live = true ;
     22 
     23     GGeo* gg = new GGeo( nullptr, live );   // picks up preexisting Opticks::Instance
     24 
     25     X4PhysicalVolume xtop(gg, top) ;  // lots of heavy lifting translation in here 
     26 
     27     gg->postDirectTranslation();
     28 
     29     return gg ;
     30 }


::

     199 void X4PhysicalVolume::init()
     200 {
     201     LOG(LEVEL) << "[" ;
     202     LOG(LEVEL) << " query : " << m_query->desc() ;
     203 
     204 
     205     convertWater();       // special casing in Geant4 forces special casing here
     206     convertMaterials();   // populate GMaterialLib
     207     convertScintillators();
     208 
     209 
     210     convertSurfaces();    // populate GSurfaceLib
     211     closeSurfaces();
     212     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVo     lume)  
     213     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     214     convertCheck();       // checking found some nodes
     215 
     216     postConvert();        // just reporting 
     217 
     218     LOG(LEVEL) << "]" ;
     219 }



