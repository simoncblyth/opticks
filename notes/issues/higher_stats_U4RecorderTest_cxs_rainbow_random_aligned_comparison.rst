higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison
========================================================================

* from :doc:`U4RecorderTest_U4Stack_stag_enum_random_alignment`


Overview : what is the purpose of random aligned comparison
-----------------------------------------------------------------

* the purpose of the random aligned comparison is to find and fix any unexpected problems (ie bugs) 
  that will manifest as array differences which cannot be explained

* also the level of difference is something that needs to be 
  reported as a characteristic of the Opticks simulation 

* so finding the levels and reasons behind differences is sufficient : the point in doing 
  the comparison is to find and investigate any unexplained differences 

* of course if the levels of difference can be reduced without costing performance 
  then could consider making changes : but typically reducing differences requires
  using double rather than float which is to be avoided if at all possible



TODO : change geometry/input photon shape to reduce edge skimmers, which look to be the largest single cause of deviation   
---------------------------------------------------------------------------------------------------------------------------


DONE : systematic presentation of deviation level : opticks.sysrap.dv using opticks.ana.array_repr_mixin and sysrap/dv.sh
----------------------------------------------------------------------------------------------------------------------------

::

    A_FOLD : /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest 
    B_FOLD : /tmp/blyth/opticks/U4RecorderTest 
    ./dv.sh   # cd ~/opticks/sysrap

                     pdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[   47,   117,  1732,  4412,  2710,   965,    16,     1,     0,     0],
                    time :        [ 2746,  5430,  1724,    96,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6404,  2937,   647,    11,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)

                     rdv :         1e-06 1e-05  0.0001 0.001  0.01   0.1    1      10     100    1000    

                     pos : array([[    4,    25,  1124,  5155,  2710,   965,    16,     1,     0,     0],
                    time :        [ 2732,  5441,  1719,   104,     4,     0,     0,     0,     0,     0],
                     mom :        [ 6388,  2953,   647,    11,     1,     0,     0,     0,     0,     0],
                     pol :        [ 9995,     1,     1,     3,     0,     0,     0,     0,     0,     0],
                      wl :        [10000,     0,     0,     0,     0,     0,     0,     0,     0,     0]], dtype=uint32)



* review what was done in old workflow ab.py and cherrypick 
* ana/ab.py not easy to cherry pick from : until have a specific need which can go hunt for, like amax::

    1286     def rpost_dv_where(self, cut):
    1287         """
    1288         :return photon indices with item deviations exceeding the cut: 
    1289         """
    1290         av = self.a.rpost()
    1291         bv = self.b.rpost()
    1292         dv = np.abs( av - bv )
    1293         return self.a.where[np.where(dv.max(axis=(1,2)) > cut) ]
    1294 

* in redoing : focus on generic handling, so can do more with less code more systematically 

A general requirement is to know the deviation profile of various quantities::

    wseq = np.where( a.seq[:,0] == b.seq[:,0] )     
    abp = np.abs( a.photon[wseq] - b.photon[wseq] )  ## for deviations to be meaningful needs to be same history  

    abp_pos  = np.amax( abp[:,0,:3], axis=1 )        ## amax of the 3 position deviations, so can operate at photon position level, not x,y,z level 
    abp_time = abp[:,0,3]
    abp_mom  = np.amax( abp[:,1,:3], axis=1 )
    abp_pol  = np.amax( abp[:,2,:3], axis=1 )

    assert abp_pos.shape == abp_time.shape == abp_mom.shape == abp_pol.shape

So it comes down to histogramming bin count frequencies of an array with lots of small values.::

   bins = np.array( [0.,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], dtype=np.float32 )  
   prof, bins2 = np.histogram( abp_pos, bins=bins )
   

DONE : Pumped up the volume to 10,000 with raindrop geometry using box factor 10. 
------------------------------------------------------------------------------------

Surprised to find the 10k are fully history aligned without any more work when including scatter from the higher stats::

    In [2]: np.where( a.seq[:,0] != b.seq[:,0] )
    Out[2]: (array([], dtype=int64),)

Substantial deviation::

    In [6]: np.abs( a.photon - b.photon ).max()
    Out[6]: 4.0538635

    In [7]: np.abs( a.record - b.record ).max()
    Out[7]: 4.0538635


    In [13]: np.where( np.abs(a.photon - b.photon) > 0.1 )
    Out[13]: 
    (array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 1, 2, 0, 0, 1]))

    In [50]: w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; w
    Out[50]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964])

    In [88]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ) ; w   ## need to unique it to avoid same photon index appearing multiple times
    Out[88]: array([ 675,  911, 1355, 1957, 2293, 2436, 2597, 4029, 5156, 5208, 7203, 7628, 7781, 8149, 8393, 9516, 9964])

    In [89]: seqhis_(a.seq[w,0])
    Out[89]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO SC BR SA',
     'TO BT BT AB',
     'TO SC SA',
     'TO BT BR BR BR BR BT SA',
     'TO BR SA',
     'TO BR SA',
     'TO BT BT AB',
     'TO BR SA',
     'TO BT SC BT SA']


more systematic look at 17/10k > 0.1 mm deviants (~1 in a thousand level) using ana/p.py:cuss 
---------------------------------------------------------------------------------------------------

::

    In [66]: w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; w
    Out[66]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964])


    In [10]: cuss(s,w)
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


::

     w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; s = a.seq[w,0] ; cuss(s,w)

In summary::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



    In [1]: cuss(a.seq[:,0])
    Out[1]: 
    CUSS([['w0', '                   TO BT BT SA', '           36045', '            8653'],
          ['w1', '                      TO BR SA', '            2237', '             691'],
          ['w2', '                TO BT BR BT SA', '          576461', '             513'],
          ['w3', '             TO BT BR BR BT SA', '         9223117', '              60'],
          ['w4', '                      TO BT AB', '            1229', '              27'],
          ['w5', '          TO BT BR BR BR BT SA', '       147569613', '              23'],
          ['w6', '                      TO SC SA', '            2157', '               9'],
          ['w7', '                TO BT BT SC SA', '          552141', '               7'],
          ['w8', '       TO BT BR BR BR BR BT SA', '      2361113549', '               4'],
          ['w9', '                TO BT SC BT SA', '          575181', '               2'],
          ['w10', '                   TO BR SC SA', '           34493', '               2'],
          ['w11', '                   TO BT BT AB', '           19661', '               2'],
          ['w12', '                   TO BT BR AB', '           19405', '               2'],
          ['w13', '             TO BT BR BT SC SA', '         8833997', '               2'],
          ['w14', '    TO BT BR BR BR BR BR BT SA', '     37777816525', '               1'],
          ['w15', '                   TO SC BR SA', '           35693', '               1'],
          ['w16', ' TO BT BR BR BR BR BR BR BT SA', '    604445064141', '               1']], dtype=object)



Summary of > 0.1 mm deviants : skimmers and absorption/scatter distance diff : these are expected float/double differences
-----------------------------------------------------------------------------------------------------------------------------

::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],          ## skimmers  
          ['w1', '                   TO BT BT AB', '           19661', '               2'],          ## absorption position
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],          ## lots of bounces 
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],          ## scatter position 
          ['w4', '                   TO SC BR SA', '           35693', '               1'],          ## scatter position 
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)  ## scatter position 



w0 : TO BR SA : > 0.1 mm deviants from 10k sample : they are all tangential grazing incidence edge skimmers
---------------------------------------------------------------------------------------------------------------

::

    In [19]: seqhis_(a.seq[w0,0])
    Out[19]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA']

These BR all end up at top ? Edge skimmer ?::

    In [12]: a.record[w0,:3,0]
    Out[12]: 
    array([[[   1.403,  -49.872, -990.   ,    0.   ],
            [   1.403,  -49.872,   -3.279,    3.292],
            [   5.126, -182.258, 1000.   ,    6.669]],

           [[  43.282,  -24.992, -990.   ,    0.   ],
            [  43.282,  -24.992,   -1.458,    3.298],
            [  93.917,  -54.23 , 1000.   ,    6.645]],

           [[ -38.393,   31.995, -990.   ,    0.   ],
            [ -38.393,   31.995,   -1.521,    3.298],
            [ -85.258,   71.05 , 1000.   ,    6.646]],

           [[ -22.29 ,   44.614, -990.   ,    0.   ],
            [ -22.29 ,   44.614,   -3.579,    3.291],
            [ -87.009,  174.153, 1000.   ,    6.674]],

           [[ -49.146,   -8.528, -990.   ,    0.   ],
            [ -49.146,   -8.528,   -3.455,    3.292],
            [-186.776,  -32.411, 1000.   ,    6.672]],

           [[  15.008,  -47.688, -990.   ,    0.   ],
            [  15.008,  -47.688,   -0.829,    3.3  ],
            [  24.977,  -79.366, 1000.   ,    6.642]],

           [[  -0.671,  -49.849, -990.   ,    0.   ],
            [  -0.671,  -49.849,   -3.824,    3.29 ],
            [  -2.756, -204.756, 1000.   ,    6.679]],

           [[ -47.523,  -15.129, -990.   ,    0.   ],
            [ -47.523,  -15.129,   -3.553,    3.291],
            [-184.473,  -58.728, 1000.   ,    6.674]],

           [[  -0.895,   49.92 , -990.   ,    0.   ],
            [  -0.895,   49.92 ,   -2.669,    3.294],
            [  -2.823,  157.42 , 1000.   ,    6.659]],

           [[  19.233,   46.065, -990.   ,    0.   ],
            [  19.233,   46.065,   -2.839,    3.294],
            [  63.329,  151.683, 1000.   ,    6.661]],

           [[  46.313,  -17.856, -990.   ,    0.   ],
            [  46.313,  -17.856,   -6.021,    3.283],
            [ 277.431, -106.965, 1000.   ,    6.74 ]]], dtype=float32)


    In [15]: a.record[w0[0],:3]  - b.record[w0[0],:3]
    Out[15]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.   ,  0.   ,  0.004,  0.   ],
        [-0.   ,  0.   ,  0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[-0.005,  0.165,  0.   , -0.   ],
        [-0.   ,  0.   ,  0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)

    In [16]: a.record[w0[1],:3]  - b.record[w0[1],:3]
    Out[16]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.   ,  0.   , -0.004, -0.   ],
        [ 0.   , -0.   , -0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]],

       [[ 0.134, -0.077, -0.   ,  0.   ],
        [ 0.   , -0.   , -0.   ,  0.   ],
        [-0.   , -0.   , -0.   ,  0.   ],
        [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)

radius of 50 does not shows its a tangent edge skimmer, just shows sphere intersect, see below need to check xy::

    In [38]: np.sqrt(np.sum(xpos*xpos,axis=1))
    Out[38]: array([ 991.261,   50.   , 1003.455], dtype=float32)

    In [65]: seqhis_(a.seq[w0,0]) 
    Out[65]: 
    ['TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA',
     'TO BR SA']

    In [20]: a.record[w0,1,0,:3]
    Out[20]: 
    array([[  1.403, -49.872,  -3.279],
           [ 43.282, -24.992,  -1.458],
           [-38.393,  31.995,  -1.521],
           [-22.29 ,  44.614,  -3.579],
           [-49.146,  -8.528,  -3.455],
           [ 15.008, -47.688,  -0.829],
           [ -0.671, -49.849,  -3.824],
           [-47.523, -15.129,  -3.553],
           [ -0.895,  49.92 ,  -2.669],
           [ 19.233,  46.065,  -2.839],
           [ 46.313, -17.856,  -6.021]], dtype=float32)

    In [22]: a.record[w0,1,0,:3] - b.record[w0,1,0,:3]  ## deviation in z of intersect 
    Out[22]: 
    array([[ 0.   ,  0.   ,  0.004],
           [ 0.   ,  0.   , -0.004],
           [ 0.   ,  0.   , -0.006],
           [ 0.   ,  0.   , -0.003],
           [ 0.   ,  0.   , -0.003],
           [ 0.   ,  0.   , -0.018],
           [ 0.   ,  0.   ,  0.003],
           [ 0.   ,  0.   ,  0.003],
           [ 0.   ,  0.   ,  0.006],
           [ 0.   ,  0.   ,  0.005],
           [ 0.   ,  0.   ,  0.002]], dtype=float32)


    In [70]: x = a.record[ww,1,0,:3]

    In [71]: np.sqrt(np.sum(x*x,axis=1))
    Out[71]: array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.], dtype=float32)


Actually the 50. does not say its an edge skimmer, any hit on the sphere will give that, need to look at xy::

    In [23]: xy = a.record[w0,1,0,:2]
    In [24]: xy
    Out[24]: 
    array([[  1.403, -49.872],
           [ 43.282, -24.992],
           [-38.393,  31.995],
           [-22.29 ,  44.614],
           [-49.146,  -8.528],
           [ 15.008, -47.688],
           [ -0.671, -49.849],
           [-47.523, -15.129],
           [ -0.895,  49.92 ],
           [ 19.233,  46.065],
           [ 46.313, -17.856]], dtype=float32)

    In [25]: np.sqrt(np.sum(xy*xy,axis=1))
    Out[25]: array([49.892, 49.979, 49.977, 49.872, 49.881, 49.993, 49.853, 49.873, 49.928, 49.919, 49.636], dtype=float32)

    In [26]: 50.-np.sqrt(np.sum(xy*xy,axis=1))
    Out[26]: array([0.108, 0.021, 0.023, 0.128, 0.119, 0.007, 0.147, 0.127, 0.072, 0.081, 0.364], dtype=float32)


Looking at the xy radius shows that these are photons hitting the sphere within around 0.1mm of its projected edge. 



w1 : TO BT BT AB : deviation all in the absorption position : known log(u_float) vs log(u_double) issue 
-----------------------------------------------------------------------------------------------------------

::

    In [9]: a.record[w1,:4] - b.record[w1,:4]
    Out[9]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   ,  0.   ,  0.   , -0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.159, -0.053, -0.417, -0.001],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]],


           [[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.   , -0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   ,  0.   , -0.   , -0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.187,  0.102, -0.422, -0.002],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)




w2 : TO BT BR BR BR BR BT SA
--------------------------------


::

    In [28]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[28]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



    In [33]: a.record[w2,:9] - b.record[w2,:9]
    Out[33]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.002,  0.002,  0.   ,  0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.001,  0.   , -0.003,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]

            [[ 0.002, -0.001, -0.001,  0.   ],
             [-0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.001, -0.001,  0.002,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.001,  0.001,  0.001,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.171, -0.   ],     ### combination of small after 6 bounces on the sphere  
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ]]]], dtype=float32)




w3 : TO BT SC BT SA : deviation starts from where the scatter happens
------------------------------------------------------------------------

::

    In [2]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w)
    Out[2]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


    In [6]: a.record[w3,:5] - b.record[w3,:5]
    Out[6]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.   , -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [ 0.   , -0.   , -0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.   , -0.   , -0.018, -0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.602,  0.219,  0.   ,  0.001],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



w4 : "TO SC BR SA" : a 1/10k > 0.1 mm deviant : small scatter position diff gets lever armed into big diff
-------------------------------------------------------------------------------------------------------------------------

::

    In [10]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w) 
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)



* HMM: this is float/double difference in handling the calculation of scattering length

* I could reduce the difference by doing the log of rand calc in double precision 
  (did that previously in old workflow) but I am inclined to now say that there is no point in doing that : 
  where the scatter point is the result of the an random throw so worrying over the exact position is pointless

::

    In [7]: seqhis_(a.seq[w4,0])
    Out[7]: ['TO SC BR SA']


Initial 0.047 mm difference in scatter position gets lever armed into a larger deviations::

    In [9]:  a.record[w4,:4] - b.record[w4,:4]
    Out[9]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.047, -0.   ],
             [ 0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.019,  0.052,  0.055,  0.   ],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.249,  0.   ,  4.054,  0.006],
             [ 0.   , -0.001,  0.003,  0.   ],
             [-0.   ,  0.   , -0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



w5 : TO SC SA : same again, difference in scattering length is cause
--------------------------------------------------------------------------

::

    In [10]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0]) ; s = a.seq[w,0] ; cuss(s,w) 
    Out[10]: 
    CUSS([['w0', '                      TO BR SA', '            2237', '              11'],
          ['w1', '                   TO BT BT AB', '           19661', '               2'],
          ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
          ['w3', '                TO BT SC BT SA', '          575181', '               1'],
          ['w4', '                   TO SC BR SA', '           35693', '               1'],
          ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)


    In [14]: a.record[w5,:3] - b.record[w5,:3]
    Out[14]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.048, -0.   ],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[-0.316, -0.15 ,  0.   , -0.001],
             [-0.   , -0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]]]], dtype=float32)



Biggest > 0.5 mm deviants : skimmer and two scatters
-------------------------------------------------------

::

    In [18]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.5 )[0]) ; w
    Out[18]: array([2436, 5156, 9964]
    In [20]: seqhis_(a.seq[w,0]) 
    Out[20]: ['TO BR SA', 'TO SC BR SA', 'TO BT SC BT SA']


