tboolean-proxy-g4evt-immediate-absorption
===============================================

Context
---------

Following major event paths refactor (and before than test geometry proxying implementation
and container resizing), g4 photon propagation is broken.

* :doc:`opticks-event-paths`
* :doc:`tboolean-with-proxylv-bringing-in-basis-solids`
* :doc:`review-test-geometry`


To create the events
-----------------------

::

   PROXYLV=17 tboolean.sh             ## compute mode
   PROXYLV=17 tboolean.sh --interop   ## propagate and visualize it 


Issue : all g4 photons are immediately absorbed without going anywhere
----------------------------------------------------------------------------


tboolean-;PROXYLV=17 tboolean-proxy-ip::

    A tboolean-proxy-17/tboolean-proxy-17/torch/  1 :  20190610-2223 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-17/evt/tboolean-proxy-17/torch/1/fdom.npy () 
    B tboolean-proxy-17/tboolean-proxy-17/torch/ -1 :  20190610-2223 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/tboolean-proxy-17/evt/tboolean-proxy-17/torch/-1/fdom.npy (recstp) 
    tboolean-proxy-17
    .                seqhis_ana  1:tboolean-proxy-17:tboolean-proxy-17   -1:tboolean-proxy-17:tboolean-proxy-17        c2        ab        ba 
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  
    0000           8ccccd      7728         0          7728.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO BT BT BT BT SA
    0001              8bd       580         0           580.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO BR SA
    0002            8cbcd       564         0           564.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BR BT SA
    0003          8ccbccd       491         0           491.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BR BT BT SA
    0004        8cccbcccd       423         0           423.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BT BR BT BT BT SA
    0005       8cccbcbccd        29         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BT BT SA
    0006         8ccbbccd        28         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BR BR BT BT SA
    0007       ccbccbcccd        26         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BT BT
    0008         8cccbbcd        26         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BR BR BT BT BT SA
    0009         8cbbcccd        24         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT BT BR BR BT SA
    0010       8ccbcbcccd        20         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BR BT BT SA
    0011              86d         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO SC SA
    0012          8cbbbcd         6         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BR BR BR BT SA
    0013        8cbbcbccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [9 ] TO BT BT BR BT BR BR BT SA
    0014       ccbbcbcccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BR BR BT BT
    0015       bcbccbcccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BT BR
    0016          8cc6ccd         4         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT SA
    0017       cbbccbcccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BT BR BT BT BR BR BT
    0018       bbccbcbccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BT BR BR
    0019          86ccccd         3         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT BT BT SC SA
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  


    ## adjust the slice to find the g4 photons, they are all under "TO AB"

    In [7]: ab.his[35:50]
    Out[7]: 
    .                seqhis_ana  1:tboolean-proxy-17:tboolean-proxy-17   -1:tboolean-proxy-17:tboolean-proxy-17        c2        ab        ba 
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  
    0035       ccbcbcbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BT BR BT BT
    0036       cbbcbccc6d         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT BT BT BR BT BR BR BT
    0037       cbcbcbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BT BR BT BR BT
    0038       ccbcbbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BR BT BR BT BT
    0039       bcbcbbbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BR BR BT BR BT BR
    0040       bbbbbcbccd         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO BT BT BR BT BR BR BR BR BR
    0041               4d         0     10000         10000.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO AB
    .                              10000     10000     19786.00/5 = 3957.20  (pval:0.000 prob:1.000)  




    In [12]: a.seqhis_ls[:5]
    Out[12]: 
    TO BT BT BT BT SA
    TO BT BT BR BT BT SA
    TO BR SA
    TO BT BT BT BT SA
    TO BT BT BT BT SA

    In [13]: b.seqhis_ls[:5]
    Out[13]: 
    TO AB
    TO AB
    TO AB
    TO AB
    TO AB

    In [14]: b.seqhis
    Out[14]: 
    A()sliced
    A([77, 77, 77, ..., 77, 77, 77], dtype=uint64)

    In [15]: np.unique(b.seqhis)
    Out[15]: 
    A()sliced
    A([77], dtype=uint64)


All B are two steps going nowhere::

    In [13]: b.rpostn(2).shape
    Out[13]: (10000, 2, 4)

    In [14]: a.rpostn(2).shape
    Out[14]: (0, 2, 4)

    In [15]: b.rpostn(2)
    Out[15]: 
    A()sliced
    A([[[  20.6922,  -63.5134, -825.8752,    0.    ],
        [  20.6922,  -63.5134, -825.8752,    0.    ]],

       [[ -48.9204,   -0.5293, -825.8752,    0.    ],
        [ -48.9204,   -0.5293, -825.8752,    0.    ]],

       [[ -74.351 ,   17.9955, -825.8752,    0.    ],
        [ -74.351 ,   17.9955, -825.8752,    0.    ]],

       ...,

       [[ -18.8272,   74.0233, -825.8752,    0.    ],
        [ -18.8272,   74.0233, -825.8752,    0.    ]],

       [[ -16.0548,   36.1925, -825.8752,    0.    ],
        [ -16.0548,   36.1925, -825.8752,    0.    ]],

       [[  28.7322,   56.8848, -825.8752,    0.    ],
        [  28.7322,   56.8848, -825.8752,    0.    ]]])




    In [2]: x = b.rpostn(2)

    In [3]: x.shape
    Out[3]: (10000, 2, 4)

    In [5]: x[:,0,2]
    Out[5]: 
    A([-825.8752, -825.8752, -825.8752, ..., -825.8752, -825.8752, -825.8752])

    In [6]: np.unique(x[:,0,2])    ## all same
    Out[6]: 
    A([-825.8752])




Hmm unexplained z-difference, perhaps a start delta to avoid being stuck on boundary ?

* hmm that might explain the peculiar photon behaviour observed in :doc:`tboolean-with-proxylv-bringing-in-basis-solids`
  with large extent proxies if the start delta was not big enough  

::

    [blyth@localhost issues]$ np.py $TMP/cg4/primary.npy -v --sli 0:10
    a :                          /tmp/blyth/location/cg4/primary.npy :        (10000, 4, 4) : f1520b5be97926aff24f10f576f0a725 : 20190610-2223 
    (10000, 4, 4)
    f32
    [[[  20.6971  -63.5045 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]

     [[ -48.9207   -0.5178 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]

     [[ -74.3543   17.9927 -903.7001    0.    ]
      [  -0.       -0.        1.        1.    ]
      [   0.       -1.        0.        0.    ]
      [   0.        0.        0.        0.    ]]




First Thing : switch on some g4 debug 
------------------------------------------

::

    PROXYLV=17 tboolean.sh --dbgrec              # this fairly useless, machinery debug 

    PROXYLV=17 tboolean.sh --dbgseqhis 0x4d      # this looks useful, dumping just "TO AB" photons which is all of them  

    PROXYLV=17 tboolean.sh --dbgseqhis 0x4d --generateoverride 5       ## restrict to 1st 5 photons


* hmm need to look into UNIVERSE_PV 


According to g4 the photons are starting in Rock and immediately get absorbed::

    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump@159] CDebug::postTrack
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@162] CDebug::dump record_id 1  origin[ -48.921-0.518-903.700]   Ori[ -48.921-0.518-903.700] 
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@168]  nstp 1
    ( 0)  TO/AB     Und   PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    [   0](Stp ;opticalphoton stepNum    1(tk ;opticalphoton tid 2 pid 0 nm    380 mm  ori[  -48.921  -0.518-903.700]  pos[    0.000   0.000   0.002]  )
      pre               UNIVERSE_PV            Rock          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock    OpAbsorption    PostStepDoItProc pos[      0.000     0.000     0.002]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     )
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@172]  npoi 0
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id        1 m_photon._badflag     0 --dbgseqhis  sas: PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@185]  seqhis               4d    TO AB                                           
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@190]  mskhis             1008    AB|TO
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@195]  seqmat               33    Rock Rock - - - - - - - - - - - - - - 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_sequence@203] CDebug::dump_sequence
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_points@229] CDeug::dump_points
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump@159] CDebug::postTrack
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@162] CDebug::dump record_id 0  origin[ 20.697-63.504-903.700]   Ori[ 20.697-63.504-903.700] 
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@168]  nstp 1
    ( 0)  TO/AB     Und   PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    [   0](Stp ;opticalphoton stepNum    1(tk ;opticalphoton tid 1 pid 0 nm    380 mm  ori[   20.697 -63.504-903.700]  pos[    0.000   0.000   0.003]  )
      pre               UNIVERSE_PV            Rock          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock    OpAbsorption    PostStepDoItProc pos[      0.000     0.000     0.003]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     )
    2019-06-10 23:07:40.307 INFO  [50323] [CRec::dump@172]  npoi 0
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id        0 m_photon._badflag     0 --dbgseqhis  sas: PRE_SAVE POST_SAVE POST_DONE LAST_POST STEP_START 
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@185]  seqhis               4d    TO AB                                           
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@190]  mskhis             1008    AB|TO
    2019-06-10 23:07:40.307 INFO  [50323] [CDebug::dump_brief@195]  seqmat               33    Rock Rock - - - - - - - - - - - - - - 
    2019-0





