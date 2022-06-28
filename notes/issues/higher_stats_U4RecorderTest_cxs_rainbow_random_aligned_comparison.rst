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


more systematic handling of the 0.1 mm deviants
-------------------------------------------------

::

     58 def ocus(w,s):
     59     """
     60     :param w: np.unique.np.where array of indices of a sub-selection eg deviants 
     61     :param s: array of seqhis with same shape as w
     62 
     63     CAUTION: this populates the builtins calling scope with w0, w1, w2, ... np.where arrays 
     64     ::
     65 
     66         w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])
     67         s = a.seq[w,0]  
     68         o, cu = ocus(w,s)
     69         print(o)
     70         print(w1)
     71     
     72     """
     73     assert w.shape == s.shape
     74     cu = cus(s)
     75     o = np.zeros( (cu.shape[0], cu.shape[1]+2), dtype=np.object )
     76     o[:,0] = list(map(lambda _:"w%d" % _ , list(range(len(cu))) ))
     77     o[:,1] = list(map(lambda _:"%30s" % _, seqhis_(cu[:,0]) ))
     78     o[:,2] = list(map(lambda _:"%16s" % _, cu[:,0] ))
     79     o[:,3] = list(map(lambda _:"%16s" % _, cu[:,1] ))
     80     if not w is None:
     81         for i in range(len(cu)):
     82             setattr( builtins, "w%d"%i, w[s == cu[i,0]] )
     83         pass
     84     pass
     85     return o, cu
     86 

::

    In [12]: o
    Out[12]: 
    array([['w0', '                      TO BR SA', '            2237', '              11'],
           ['w1', '                   TO BT BT AB', '           19661', '               2'],
           ['w2', '       TO BT BR BR BR BR BT SA', '      2361113549', '               1'],
           ['w3', '                TO BT SC BT SA', '          575181', '               1'],
           ['w4', '                   TO SC BR SA', '           35693', '               1'],
           ['w5', '                      TO SC SA', '            2157', '               1']], dtype=object)




TO BT BT AB : deviation all in the absorption position : known log(u_float) vs log(u_double) issue 
-----------------------------------------------------------------------------------------------------

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




0.5 mm deviants
--------------------

::

    In [18]: w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.5 )[0]) ; w
    Out[18]: array([2436, 5156, 9964]
    In [20]: seqhis_(a.seq[w,0]) 
    Out[20]: ['TO BR SA', 'TO SC BR SA', 'TO BT SC BT SA']



Look at 2/10k "TO BT BT AB"
-----------------------------

::







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


radius of 50 does not shows its a tangent edge skimmer, just shows sphere intersect, see below need to check xy::

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


Actually the 50. does not say its an edge skimmer, any hit on the sphere will give that, need to look at xy::

    In [100]: xy = a.record[ww,1,0,:2]
    In [101]: xy
    Out[101]: 
    array([[  1.403, -49.872],
           [ 43.282, -24.992],
           [-38.393,  31.995],
           [-38.393,  31.995],
           [-22.29 ,  44.614],
           [-49.146,  -8.528],
           [ 15.008, -47.688],
           [ 15.008, -47.688],
           [ -0.671, -49.849],
           [-47.523, -15.129],
           [ -0.895,  49.92 ],
           [ 19.233,  46.065],
           [ 46.313, -17.856]], dtype=float32)

    In [102]: np.sqrt(np.sum(xy*xy,axis=1))
    Out[102]: array([49.892, 49.979, 49.977, 49.977, 49.872, 49.881, 49.993, 49.993, 49.853, 49.873, 49.928, 49.919, 49.636], dtype=float32)


Looking at the xy radius shows that these are photons hitting the sphere within around 0.1mm of its projected edge. 


TO SC BR SA
--------------

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



