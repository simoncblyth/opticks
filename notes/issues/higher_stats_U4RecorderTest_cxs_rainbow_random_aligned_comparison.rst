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


Pumped up the volume to 10,000 with raindrop geometry using box factor 10. 
----------------------------------------------------------------------------

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


0.5 mm deviants
--------------------

::

    In [18]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.5 )[0]) ; w
    Out[18]: array([2436, 5156, 9964]
    In [20]: seqhis_(a.seq[w,0]) 
    Out[20]: ['TO BR SA', 'TO SC BR SA', 'TO BT SC BT SA']


Take a look at the "TO SC BR SA" : a 1/10k > 0.1 mm deviant : small scatter position diff gets lever armed into big diff
---------------------------------------------------------------------------------------------------------------------------

* HMM: this is float/double difference in handling the calculation of scattering length

* I could reduce the difference by doing the log of rand calc in double precision 
  (did that previously in old workflow) but I am inclined to now say that there is no point in doing that : 
  where the scatter point is the result of the an random throw so worrying over the exact position is pointless

::

    In [92]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ) ; w
    Out[92]: array([ 675,  911, 1355, 1957, 2293, 2436, 2597, 4029, 5156, 5208, 7203, 7628, 7781, 8149, 8393, 9516, 9964])

    In [93]: w2 = w[ a.seq[w,0] == 35693 ] ; w2
    Out[93]: array([5156])

    In [94]: seqhis_(a.seq[w2,0])
    Out[94]: ['TO SC BR SA']


    In [95]: a.record[w2,:4]
    Out[95]: 
    array([[[[  -2.599,   22.393, -990.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ],
             [   0.993,    0.115,    0.   ,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]],

            [[  -2.599,   22.393,  -59.126,    3.106],
             [  -0.161,    0.451,    0.878,    0.   ],
             [  -0.982,   -0.165,   -0.096,  501.   ],
             [   0.   ,    0.   ,   -0.   ,    0.   ]],

            [[  -6.946,   34.552,  -35.467,    3.196],
             [  -0.241,    0.85 ,    0.468,    0.   ],
             [   0.963,    0.27 ,    0.006,  501.   ],
             [   0.   ,    0.   ,   -0.   ,    0.   ]],

            [[-281.197, 1000.   ,  496.732,    6.986],
             [  -0.241,    0.85 ,    0.468,    0.   ],
             [   0.963,    0.27 ,    0.006,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]]]], dtype=float32)

    In [96]: b.record[w2,:4]
    Out[96]: 
    array([[[[  -2.599,   22.393, -990.   ,    0.   ],
             [   0.   ,    0.   ,    1.   ,    0.   ],
             [   0.993,    0.115,    0.   ,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]],

            [[  -2.599,   22.393,  -59.079,    3.106],
             [  -0.161,    0.451,    0.878,    0.   ],
             [  -0.982,   -0.165,   -0.096,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]],

            [[  -6.927,   34.499,  -35.522,    3.196],
             [  -0.242,    0.851,    0.466,    0.   ],
             [   0.963,    0.27 ,    0.006,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]],

            [[-280.948, 1000.   ,  492.678,    6.98 ],
             [  -0.242,    0.851,    0.466,    0.   ],
             [   0.963,    0.27 ,    0.006,  501.   ],
             [   0.   ,    0.   ,    0.   ,    0.   ]]]], dtype=float32)


    In [97]: a.record[w2,:4] - b.record[w2,:4]
    Out[97]: 
    array([[[[ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   , -0.   ,  0.   ]],

            [[ 0.   ,  0.   , -0.047, -0.   ],     ## initial 0.047 mm difference in scatter position gets lever armed into a larger deviations 
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



Take a look at the "TO BR SA" > 0.1 mm deviants from 10k sample : they are all tangential grazing incidence edge skimmers
---------------------------------------------------------------------------------------------------------------------------

Huh BR that ends up at top ? Edge skimmer ?::


    In [24]: a.record[w[0],:3]
    Out[24]: 
    array([[[  15.008,  -47.688, -990.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [  -0.954,   -0.3  ,    0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  15.008,  -47.688,   -0.829,    3.3  ],
            [   0.01 ,   -0.032,    0.999,    0.   ],
            [   0.954,    0.3  ,   -0.   ,  501.   ],
            [   0.   ,    0.   ,   -0.   ,    0.   ]],

           [[  24.977,  -79.366, 1000.   ,    6.642],
            [   0.01 ,   -0.032,    0.999,    0.   ],
            [   0.954,    0.3  ,   -0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [25]: b.record[w[0],:3]
    Out[25]: 
    array([[[  15.008,  -47.688, -990.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [  -0.954,   -0.3  ,    0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  15.008,  -47.688,   -0.811,    3.3  ],
            [   0.01 ,   -0.031,    0.999,    0.   ],
            [   0.954,    0.3  ,   -0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  24.761,  -78.679, 1000.   ,    6.641],
            [   0.01 ,   -0.031,    0.999,    0.   ],
            [   0.954,    0.3  ,   -0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)


radius of 50 shows its a tangent edge skimmer::

    In [38]: np.sqrt(np.sum(xpos*xpos,axis=1))
    Out[38]: array([ 991.261,   50.   , 1003.455], dtype=float32)


::

    In [66]: w = np.where( np.abs(a.photon - b.photon) > 0.1 )[0] ; w
    Out[66]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 5156, 5156, 5208, 5208, 7203, 7203, 7628, 7781, 8149, 8393, 8393, 8393, 9516, 9964, 9964])

    In [64]: ww = w[ a.seq[w,0] == 2237 ] ; ww    ## select the "TO BR SA" from the deviants 
    Out[64]: array([ 675,  911, 1355, 1355, 1957, 2293, 2436, 2436, 2597, 4029, 7781, 8149, 9516])

    In [65]: seqhis_(a.seq[ww,0]) 
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


    In [68]: a.record[ww,1,0,:3]
    Out[68]: 
    array([[  1.403, -49.872,  -3.279],
           [ 43.282, -24.992,  -1.458],
           [-38.393,  31.995,  -1.521],
           [-38.393,  31.995,  -1.521],
           [-22.29 ,  44.614,  -3.579],
           [-49.146,  -8.528,  -3.455],
           [ 15.008, -47.688,  -0.829],
           [ 15.008, -47.688,  -0.829],
           [ -0.671, -49.849,  -3.824],
           [-47.523, -15.129,  -3.553],
           [ -0.895,  49.92 ,  -2.669],
           [ 19.233,  46.065,  -2.839],
           [ 46.313, -17.856,  -6.021]], dtype=float32)

    In [72]: b.record[ww,1,0,:3]
    Out[72]: 
    array([[  1.403, -49.872,  -3.283],
           [ 43.282, -24.992,  -1.454],
           [-38.393,  31.995,  -1.515],
           [-38.393,  31.995,  -1.515],
           [-22.29 ,  44.614,  -3.576],
           [-49.146,  -8.528,  -3.452],
           [ 15.008, -47.688,  -0.811],
           [ 15.008, -47.688,  -0.811],
           [ -0.671, -49.849,  -3.827],
           [-47.523, -15.129,  -3.556],
           [ -0.895,  49.92 ,  -2.675],
           [ 19.233,  46.065,  -2.844],
           [ 46.313, -17.856,  -6.023]], dtype=float32)


    In [70]: x = a.record[ww,1,0,:3]

    In [71]: np.sqrt(np.sum(x*x,axis=1))
    Out[71]: array([50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.], dtype=float32)



::

    In [46]: seqhis_(a.seq[w[1],0])
    Out[46]: 'TO SC BR SA'

    In [47]: seqhis_(b.seq[w[1],0])
    Out[47]: 'TO SC BR SA'


    In [41]: a.record[w[1],:4]
    Out[41]: 
    array([[[  -2.599,   22.393, -990.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.993,    0.115,    0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  -2.599,   22.393,  -59.126,    3.106],
            [  -0.161,    0.451,    0.878,    0.   ],
            [  -0.982,   -0.165,   -0.096,  501.   ],
            [   0.   ,    0.   ,   -0.   ,    0.   ]],

           [[  -6.946,   34.552,  -35.467,    3.196],
            [  -0.241,    0.85 ,    0.468,    0.   ],
            [   0.963,    0.27 ,    0.006,  501.   ],
            [   0.   ,    0.   ,   -0.   ,    0.   ]],

           [[-281.197, 1000.   ,  496.732,    6.986],
            [  -0.241,    0.85 ,    0.468,    0.   ],
            [   0.963,    0.27 ,    0.006,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [42]: b.record[w[1],:4]
    Out[42]: 
    array([[[  -2.599,   22.393, -990.   ,    0.   ],
            [   0.   ,    0.   ,    1.   ,    0.   ],
            [   0.993,    0.115,    0.   ,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  -2.599,   22.393,  -59.079,    3.106],
            [  -0.161,    0.451,    0.878,    0.   ],
            [  -0.982,   -0.165,   -0.096,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  -6.927,   34.499,  -35.522,    3.196],
            [  -0.242,    0.851,    0.466,    0.   ],
            [   0.963,    0.27 ,    0.006,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[-280.948, 1000.   ,  492.678,    6.98 ],
            [  -0.242,    0.851,    0.466,    0.   ],
            [   0.963,    0.27 ,    0.006,  501.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)



