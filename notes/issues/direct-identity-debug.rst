direct-identity-debug
=======================


Still problem with analytic identity.::

    202 guint4 GVolume::getIdentity()
    203 {
    204     unsigned node_index = m_index ;
    205 
    206     //unsigned identity_index = getSensorSurfaceIndex() ;   
    207     unsigned identity_index = m_copyNumber  ;
    208 
    209     // surprised to get this in the global 
    210     //if(identity_index > 300000 ) std::raise(SIGINT); 
    211 
    212     return guint4(
    213                    node_index,
    214                    getMeshIndex(),
    215                    m_boundary,
    216                    identity_index
    217                  );
    218 }


Dumping identity from random photon_id 100,1000,..::

    P=100 okt

    2020-07-23 17:34:32.468 NONE  [318343] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-23 17:34:32.468 INFO  [318343] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 100 0 0) -
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    2020-07-23 17:34:32.751 INFO  [318343] [OPropagator::launch@275] LAUNCH DONE
    2020-07-23 17:34:32.751 INFO  [318343] [OPropagator::launch@277] 0 : (0;11235,1) 


    P=1000 okt

    2020-07-23 17:35:40.539 INFO  [320174] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 1000 0 0) -
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       21        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        2 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       22        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       24        0 )
    2020-07-23 17:35:40.829 INFO  [320174] [OPropagator::launch@275] LAUNCH DONE
    2020-07-23 17:35:40.829 INFO  [320174] [OPropagator::launch@277] 0 : (0;11235,1) 


    P=1001 okt

    2020-07-23 17:36:48.356 INFO  [322052] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 1001 0 0) -
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       21        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        2 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       22        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       24        0 )
    2020-07-23 17:36:48.642 INFO  [322052] [OPropagator::launch@275] LAUNCH DONE
    2020-07-23 17:36:48.642 INFO  [322052] [OPropagator::launch@277] 0 : (0;11235,1) 

    P=1100 okt 

    2020-07-23 17:38:07.681 INFO  [324230] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 1100 0 0) -
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    2020-07-23 17:38:07.968 INFO  [324230] [OPropagator::launch@275] LAUNCH DONE



Stomp on the identity debug flags with photon_id, so can know the pindex to use for photons
destined to be hits.


    2020-07-23 18:04:22.218 INFO  [367810] [BTimes::dump@177] OPropagator::launch
                    launch001                 0.283674
    2020-07-23 18:04:22.218 INFO  [367810] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-07-23 18:04:22.259 INFO  [367810] [NPho::Dump@141] OEvent::downloadHitsCompute --dumphit,post,flgs ox Y
    2020-07-23 18:04:22.259 INFO  [367810] [NPho::dump@178] NPho::dump desc NPho 3014,4,4 numPhotons 3014
     i       0 post ( -15591.607 -8364.066 -7656.938     102.498) flgs     -27       0      15    2114
     i       1 post (  13564.103-12817.953  5029.110     125.696) flgs     -27       0      18    2130
     i       2 post (   9610.473  7584.673-14902.993     101.717) flgs     -25       0      25    2114
     i       3 post (  -4964.778-13531.276-12794.304     153.563) flgs     -27       0      26    2146
     i       4 post (   9348.926 12486.271-11332.037     112.246) flgs     -25       0      30    2162
     i       5 post (  15214.401  5903.624-10432.022     104.841) flgs     -25       0      32    2130
     i       6 post (    145.920 -1724.041-19270.643     124.687) flgs     -25       0      34    2146
     i       7 post (  -6939.222-15935.734  8456.948     101.837) flgs     -27       0      37    2114
     i       8 post (  -3224.126  3235.120 18795.668     108.261) flgs     -25       0      38    2114
     i       9 post (   -916.787 17415.383 -8207.887     128.150) flgs     -25       0      42    2130
     i      10 post (   -286.610-19286.537  1081.641     142.147) flgs     -25       0      44    2146
     i      11 post (   1988.733 10892.303-15881.978      98.537) flgs     -27       0      52    2114
     i      12 post (   4930.103-12212.612 14198.692      99.736) flgs     -27       0      58    2114
     i      13 post (  17628.059 -1190.079 -7688.262      98.650) flgs     -27       0      62    2114
     i      14 post ( -12561.900 12846.752  7191.808     108.594) flgs     -25       0      63    2130



::

    [blyth@localhost 1]$ pwd
    /tmp/blyth/opticks/OKTest/evt/g4live/torch/1
    [blyth@localhost 1]$ inp *.npy
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    a :                                                       ox.npy :        (11235, 4, 4) : 056762b49b66ff91fdc3b7fdbf22476c : 20200723-1810 
    b :                                                       ph.npy :        (11235, 1, 2) : 6dcf3c1fb57a331d03eb25c561f6360b : 20200723-1810 
    c :                                                       ps.npy :        (11235, 1, 4) : e12613446f4ffd14daa1909379f30060 : 20200723-1810 
    d :                                                       rs.npy :    (11235, 10, 1, 4) : 0c01bd3ea3ace8acb2041a11dd4030d2 : 20200723-1810 
    e :                                                       rx.npy :    (11235, 10, 2, 4) : d70543f6efefb5cd907429cbb2130605 : 20200723-1810 
    f :                                                       ht.npy :         (3014, 4, 4) : c64681ea84c2fed7b96a591dc82f7329 : 20200723-1810 
    g :                                     OpticksProfileLabels.npy :             (72, 64) : b747384004fa0062424866b0de484932 : 20200723-1810 
    h :                                           OpticksProfile.npy :              (72, 4) : bfad2fc2efafb0f1cd26d34929afe89d : 20200723-1810 
    i :                                                       gs.npy :           (64, 6, 4) : 9b9d2350fd8109de0e994d7b9b9cbf9f : 20200723-1810 
    j :                                                     fdom.npy :            (3, 1, 4) : 68063aaa199eb13c7d10b657d1f5074a : 20200723-1810 
    k :                                                     idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20200723-1810 
    l :                                  OpticksProfileAccLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20200723-1810 
    m :                                        OpticksProfileAcc.npy :               (1, 4) : 7bfd9c151baf10b62d8bf10f6433f993 : 20200723-1810 
    n :                                  OpticksProfileLisLabels.npy :              (1, 64) : 3637cf25a4163be8a5dc893fb8e1dd43 : 20200723-1810 
    o :                                        OpticksProfileLis.npy :                 (1,) : 0dc14ebe8a91f9145744c27b2fdea8b3 : 20200723-1810 

    In [1]: 


Hmm howabout writing a photon identity buffer ?

    In [3]: f.view(np.int32)[:,3]
    Out[3]: 
    array([[  -27,     0,    15,  2114],
           [  -27,     0,    18,  2130],
           [  -25,     0,    25,  2114],
           ...,
           [  -29,     0, 11224,  2146],
           [  -25,     0, 11230,  2162],
           [  -25,     0, 11232,  2162]], dtype=int32)

    In [4]:                                                                      


    In [21]: np.unique(f.view(np.int32)[:,3,3], return_counts=True)
    Out[21]: 
    (array([2113, 2114, 2129, 2130, 2146, 2161, 2162, 3170], dtype=int32),
     array([   2, 1415,   17,  499,  846,   10,  224,    1]))

    In [22]: co,cn = np.unique(f.view(np.int32)[:,3,3], return_counts=True)

    In [12]: from opticks.ana.hismask import HisMask

    In [13]: hm = HisMask()

    In [25]: map(lambda _:hm.label(int(_)), co )     ## all hits have SD, as they must with default hitmask 
    Out[25]: 
    ['BT|SD|CK',
     'BT|SD|SI',
     'BT|SD|RE|CK',
     'BT|SD|RE|SI',
     'BT|SD|SC|SI',
     'BT|SD|SC|RE|CK',
     'BT|SD|SC|RE|SI',
     'BT|BR|SD|SC|SI']


    In [35]: hit_id = f.view(np.int32)[:,3,2]

    In [36]: hit_id
    Out[36]: array([   15,    18,    25, ..., 11224, 11230, 11232], dtype=int32)

    In [37]: hit_id.shape
    Out[37]: (3014,)


    In [27]: b.shape
    Out[27]: (11235, 1, 2)

    In [30]: seqhis = b[:,0,0]




    In [5]: from opticks.ana.histype import HisType

    In [6]: af = HisType()
    INFO:opticks.ana.enum:parsing $OPTICKS_PREFIX/include/OpticksCore/OpticksPhoton.h 
    INFO:opticks.ana.enum:path expands to /home/blyth/local/opticks/include/OpticksCore/OpticksPhoton.h 


    In [39]: af.label(seqhis[hit_id[0]])
    Out[39]: 'SI BT BT BT BT BT BT SD'

    In [40]: af.label(seqhis[hit_id[1]])
    Out[40]: 'SI RE BT BT BT BT BT BT SD'

    In [41]: af.label(seqhis[hit_id[2]])
    Out[41]: 'SI BT BT BT BT BT BT SD'

    In [42]: af.label(seqhis[hit_id[3]])
    Out[42]: 'SI SC SC BT BT BT BT BT BT SD'





Continue locally
------------------

::

    scp -r P:/tmp/blyth/opticks/OKTest/evt/g4live/torch/1 /tmp/blyth/opticks/OKTest/evt/g4live/torch/1


~/.opticks_setup::

    76 export OPTICKS_ANA_DEFAULTS="det=g4live,cat=g4live,src=torch,tag=1,pfx=OKTest"


    epsilon:ana blyth$ ip

    In [1]: run evt.py
    [2020-07-23 12:08:54,116] p33322 {opticks_args        :main.py   :112} INFO     - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'g4live', 'pfx': 'OKTest', 'cat': 'g4live'} 
    [2020-07-23 12:08:54,119] p33322 {legacy_init         :env.py    :185} WARNING  - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2020-07-23 12:08:54,120] p33322 {__init__            :evt.py    :215} INFO     - [ ? 
    [2020-07-23 12:08:54,120] p33322 {__init__            :metadata.py:63} INFO     - path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/DeltaTime.ini 
    [2020-07-23 12:08:54,120] p33322 {__init__            :metadata.py:81} INFO     - path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/OpticksEvent_launch.ini does not exist 
    [2020-07-23 12:08:54,120] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/fdom.npy size 128 
    [2020-07-23 12:08:54,121] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/idom.npy size 96 
    [2020-07-23 12:08:54,125] p33322 {__init__            :enum.py   :42} INFO     - parsing $OPTICKS_PREFIX/include/OpticksCore/OpticksPhoton.h 
    [2020-07-23 12:08:54,125] p33322 {__init__            :enum.py   :44} INFO     - path expands to /usr/local/opticks/include/OpticksCore/OpticksPhoton.h 
    [2020-07-23 12:08:54,126] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/gs.npy size 6224 
    [2020-07-23 12:08:54,127] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ox.npy size 719120 
    [2020-07-23 12:08:54,135] p33322 {check_ox_fdom       :evt.py    :578} WARNING  -  t :   0.000 1200.000 : tot 11235 over 1 0.000  under 0 0.000 : mi      0.743 mx   1336.274  
    [2020-07-23 12:08:54,138] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ht.npy size 192976 
    [2020-07-23 12:08:54,139] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/rx.npy size 1797696 
    [2020-07-23 12:08:54,140] p33322 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ph.npy size 179840 
    [2020-07-23 12:08:54,214] p33322 {__init__            :evt.py    :275} INFO     - ] ? 
    noshortname?
    .                            1:g4live:OKTest 
    .                              11235         1.00 
    0000               42        0.148        1662        [2 ] SI AB
    0001         7cccccc2        0.116        1300        [8 ] SI BT BT BT BT BT BT SD
    0002        7cccccc62        0.053         597        [9 ] SI SC BT BT BT BT BT BT SD
    0003         8cccccc2        0.052         585        [8 ] SI BT BT BT BT BT BT SA
    0004              452        0.038         427        [3 ] SI RE AB
    0005              462        0.038         427        [3 ] SI SC AB
    0006        7cccccc52        0.032         362        [9 ] SI RE BT BT BT BT BT BT SD
    0007        8cccccc62        0.024         270        [9 ] SI SC BT BT BT BT BT BT SA
    0008       7cccccc662        0.018         201        [10] SI SC SC BT BT BT BT BT BT SD
    0009        8cccccc52        0.015         173        [9 ] SI RE BT BT BT BT BT BT SA
    0010       7cccccc652        0.014         153        [10] SI RE SC BT BT BT BT BT BT SD
    0011               41        0.013         145        [2 ] CK AB
    0012       cccccc6662        0.012         139        [10] SI SC SC SC BT BT BT BT BT BT
    0013             4662        0.012         132        [4 ] SI SC SC AB
    0014             4652        0.011         123        [4 ] SI RE SC AB
    0015       cccccccc62        0.011         122        [10] SI SC BT BT BT BT BT BT BT BT
    0016             4552        0.010         117        [4 ] SI RE RE AB
    0017       cccccc6652        0.010         111        [10] SI RE SC SC BT BT BT BT BT BT
    0018       7cccccc552        0.010         111        [10] SI RE RE BT BT BT BT BT BT SD
    0019           4cccc2        0.009         101        [6 ] SI BT BT BT BT AB
    .                              11235         1.00 

    In [2]: 

    In [12]: a.rpost_(slice(0,10)).shape
    Out[12]: (11235, 10, 4)

    In [14]: rp = a.rpost_(slice(0,10))

    In [16]: rp[15]
    Out[16]: 
    A([[    60.4266,    113.5289,   -419.3243,      3.6256],
       [-14304.6358,  -7666.86  ,  -7064.4246,     95.1811],
       [-14401.6846,  -7718.131 ,  -7110.2023,     95.8037],
       [-15555.2843,  -8344.3709,  -7639.3933,    102.2858],
       [-15579.0887,  -8357.1886,  -7650.38  ,    102.4323],
       [-15584.582 ,  -8360.8509,  -7654.0422,    102.4689],
       [-15586.4132,  -8360.8509,  -7654.0422,    102.4689],
       [-15591.9065,  -8364.5131,  -7657.7044,    102.5056],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ]])


    In [4]: a.bd[:20]
    Out[4]: 
    A([[ 17,  16,  16,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  17,  17,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  17,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  16,  15,  14,  13,  13,   0,   0,   0,   0],
       [ 17,  17,  16,  16,   0,   0,   0,   0,   0,   0],
       [ 17,  17,  17,  17,  17,   0,   0,   0,   0,   0],
       [ 17,  17,  17,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  17,  17,  17,  16, -22,  22, -22, -16, -16],
       [ 17,  17,  17,  17,   0,   0,   0,   0,   0,   0],
       [ 17,  17,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  16,  15,  14, -33, -16,  16, -23, -35, -35],
       [ 17,  17,  17,  16, -22, -16, -16,   0,   0,   0],
       [ 17,  17,  16,  16, -16, -20, -16,  16, -22, -22],
       [ 17,  17,  17,  17,  17,   0,   0,   0,   0,   0],
       [ 17,  17,  17,  17,   0,   0,   0,   0,   0,   0],

       [ 17,  16, -22, -16,  16, -23, -27, -27,   0,   0],

       [ 17,  17,  17,  17,  17,  16, -22, -16,  16,  16],
       [ 17,  17,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 17,  17,  16, -22, -16,  16, -23, -27, -27,   0],
       [ 17,  17,  17,  17,  16, -22, -16,  22, -22, -22]], dtype=int8)


    ## +ve bnd are from inside with photon direction in same hemi as surface normal 
    ##   1-based
    ##    17 : Acrylic///LS 
    ##    16 : Water///Acrylic 
    ##    22 : Water///Water
    ##    23 : Water///Pyrex 
    ##    27 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 



Notice material inconsistency between g4live and from GDML

* :doc:`material-inconsistency.rst`


Make it easier to load evt.py and get to hit id
--------------------------------------------------


::

    epsilon:~ blyth$ t ip
    ip () 
    { 
        local arg1=${1:-evt.py};
        shift;
        ipython -i -- $(which $arg1) $*
    }
    epsilon:~ blyth$ ip
    [2020-07-23 19:00:22,668] p50949 {opticks_args        :main.py   :112} INFO     - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'g4live', 'pfx': 'OKTest', 'cat': 'g4live'} 
    [2020-07-23 19:00:22,671] p50949 {legacy_init         :env.py    :185} WARNING  - legacy_init : OPTICKS_KEY envvar deleted for legacy running, unset IDPATH to use direct_init
    [2020-07-23 19:00:22,671] p50949 {__init__            :evt.py    :215} INFO     - [ ? 
    [2020-07-23 19:00:22,671] p50949 {__init__            :metadata.py:63} INFO     - path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/DeltaTime.ini 
    [2020-07-23 19:00:22,671] p50949 {__init__            :metadata.py:81} INFO     - path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/OpticksEvent_launch.ini does not exist 
    [2020-07-23 19:00:22,672] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/fdom.npy size 128 
    [2020-07-23 19:00:22,672] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/idom.npy size 96 
    [2020-07-23 19:00:22,676] p50949 {__init__            :enum.py   :42} INFO     - parsing $OPTICKS_PREFIX/include/OpticksCore/OpticksPhoton.h 
    [2020-07-23 19:00:22,676] p50949 {__init__            :enum.py   :44} INFO     - path expands to /usr/local/opticks/include/OpticksCore/OpticksPhoton.h 
    [2020-07-23 19:00:22,677] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/gs.npy size 6224 
    [2020-07-23 19:00:22,677] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ox.npy size 719120 
    [2020-07-23 19:00:22,687] p50949 {check_ox_fdom       :evt.py    :578} WARNING  -  t :   0.000 1200.000 : tot 11235 over 1 0.000  under 0 0.000 : mi      0.743 mx   1336.274  
    [2020-07-23 19:00:22,691] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ht.npy size 192976 
    [2020-07-23 19:00:22,692] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/rx.npy size 1797696 
    [2020-07-23 19:00:22,693] p50949 {load_               :nload.py  :276} INFO     -  path /tmp/blyth/opticks/OKTest/evt/g4live/torch/1/ph.npy size 179840 
    [2020-07-23 19:00:22,766] p50949 {__init__            :evt.py    :275} INFO     - ] ? 
    noshortname?
    .                            1:g4live:OKTest 
    .                              11235         1.00 
    0000               42        0.148        1662        [2 ] SI AB
    0001         7cccccc2        0.116        1300        [8 ] SI BT BT BT BT BT BT SD
    0002        7cccccc62        0.053         597        [9 ] SI SC BT BT BT BT BT BT SD
    0003         8cccccc2        0.052         585        [8 ] SI BT BT BT BT BT BT SA
    0004              452        0.038         427        [3 ] SI RE AB
    0005              462        0.038         427        [3 ] SI SC AB
    0006        7cccccc52        0.032         362        [9 ] SI RE BT BT BT BT BT BT SD
    0007        8cccccc62        0.024         270        [9 ] SI SC BT BT BT BT BT BT SA
    0008       7cccccc662        0.018         201        [10] SI SC SC BT BT BT BT BT BT SD
    0009        8cccccc52        0.015         173        [9 ] SI RE BT BT BT BT BT BT SA
    0010       7cccccc652        0.014         153        [10] SI RE SC BT BT BT BT BT BT SD
    0011               41        0.013         145        [2 ] CK AB
    0012       cccccc6662        0.012         139        [10] SI SC SC SC BT BT BT BT BT BT
    0013             4662        0.012         132        [4 ] SI SC SC AB
    0014             4652        0.011         123        [4 ] SI RE SC AB
    0015       cccccccc62        0.011         122        [10] SI SC BT BT BT BT BT BT BT BT
    0016             4552        0.010         117        [4 ] SI RE RE AB
    0017       cccccc6652        0.010         111        [10] SI RE SC SC BT BT BT BT BT BT
    0018       7cccccc552        0.010         111        [10] SI RE RE BT BT BT BT BT BT SD
    0019           4cccc2        0.009         101        [6 ] SI BT BT BT BT AB
    .                              11235         1.00 

    In [1]: a.htid
    Out[1]: A([   15,    18,    25, ..., 11224, 11230, 11232], dtype=int32)









::

    epsilon:ana blyth$ ./blib.py $GC
     nbnd  35 nmat  16 nsur  20 
      0 :   1 : Galactic///Galactic 
      1 :   2 : Galactic///Rock 
      2 :   3 : Rock///Air 
      3 :   4 : Air///Air 
      4 :   5 : Air///LS 
      5 :   6 : Air///Steel 
      6 :   7 : Air///Tyvek 
      7 :   8 : Air///Aluminium 
      8 :   9 : Aluminium///Adhesive 
      9 :  10 : Adhesive///TiO2Coating 
     10 :  11 : TiO2Coating///Scintillator 
     11 :  12 : Rock///Tyvek 
     12 :  13 : Tyvek///vetoWater 
     13 :  14 : vetoWater/CDTyvekSurface//Tyvek 
     14 :  15 : Tyvek///Water 
     15 :  16 : Water///Acrylic 
     16 :  17 : Acrylic///LS 
     17 :  18 : LS///Acrylic 
     18 :  19 : LS///PE_PA 
     19 :  20 : Water///Steel 
     20 :  21 : Water///PE_PA 
     21 :  22 : Water///Water 
     22 :  23 : Water///Pyrex 
     23 :  24 : Pyrex///Pyrex 
     24 :  25 : Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum 
     25 :  26 : Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum 
     26 :  27 : Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum 
     27 :  28 : Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum 
     28 :  29 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum 
     29 :  30 : Pyrex//PMT_3inch_absorb_logsurf1/Vacuum 
     30 :  31 : Water///LS 
     31 :  32 : Water/Steel_surface/Steel_surface/Steel 
     32 :  33 : vetoWater///Water 
     33 :  34 : Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum 
     34 :  35 : Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum 
    epsilon:ana blyth$ 




In intersect have identity, but that doesnt make it through to the closest hit

::

    2020-07-24 02:15:07.423 NONE  [235041] [OPropagator::setSize@152]  width 11235 height 1
    2020-07-24 02:15:07.423 NONE  [235041] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-24 02:15:07.423 INFO  [235041] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 15 0 0) -
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       16        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       16        0 )
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       15        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   3 identity (  142635      33      23   12397 ) offset    21117  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   5 identity (  142637      32      27   12397 ) offset    21119  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       21        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       21        0 )
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   3 identity (  142635      33      23   12397 ) offset    21117  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   5 identity (  142637      32      27   12397 ) offset    21119  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       15        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   3 identity (  142635      33      23   12397 ) offset    21117  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   5 identity (  142637      32      27   12397 ) offset    21119  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       15        2) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        2 )
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   3 identity (  142635      33      23   12397 ) offset    21117  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   5 identity (  142637      32      27   12397 ) offset    21119  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       22        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       22        0 )
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   3 identity (  142635      33      23   12397 ) offset    21117  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // intersect_analysic.cu:intersect WITH_PRINT_IDENTITY_INTERSECT repeat_index 3 instance_index 3519 primitive_count   6 primIdx   5 identity (  142637      32      27   12397 ) offset    21119  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       26        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       26        0 )
    2020-07-24 02:15:07.724 INFO  [235041] [OPropagator::launch@275] LAUNCH DONE
    2020-07-24 02:15:07.724 INFO  [235041] [OPropagator::launch@277] 0 : (0;11235,1) 
    2020-07-24 02:15:07.724 INFO  [235041] [BTimes::dump@177] OPropagator::launch



4.1.4 Attribute variables (6.5 manual p47)
----------------------------------------------

In addition to the semantics provided by OptiX, variables may also be declared
with user-defined semantics called attributes. Unlike built-in semantics, the
value of variables declared in this way must be managed by the programmer. 
Attribute variables provide a mechanism for communicating data between 
the intersection program and the shading programs (for example, surface normals 
and texture coordinates). Attribute variables may only be written in an 
intersection program between calls to rtPotentialIntersection and rtReportIntersection. 
Although OptiX may not find all object intersections in order along the ray, the value of the
attribute variable is guaranteed to reflect the value at the closest
intersection at the time that the closest-hit program is invoked.

**Note: Because intersections may not be found in order, programs should use
attribute variables (as opposed to the ray payload) to communicate information
about the local hit point between intersection and shading programs.**

The following example declares an attribute variable of type float3 named
normal. The semantic association of the attribute is specified with the
user-defined name normal_vec. *This name is arbitrary, and is the link between
the variable declared here and another variable declared in the closest-hit
program. The two attribute variables need not have the same name as long as
their attribute names match.*

Listing 4.7::

    rtDeclareVariable( float3, normal, attribute normal_vec, );
  

Recipe:

1. duplicate declarations in intersect and closest hit programs 

   * they must have same attribute name (eg normal_vec)
   * they can have same attribute variable name (eg normal)

 
::

    epsilon:cu blyth$ grep instance_identity *.*
    GeometryTriangles.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    TriangleMesh.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    intersect_analytic.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    material1_propagate.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    material1_radiance.cu:rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );
    sphere.cu:rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
    epsilon:cu blyth$ 


intersect_analytic.cu::

     87 rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
     88 rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
     89 rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

material1_propagate.cu::

     26 rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
     27 rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );



identity not getting from the intersect to the closest hit::


    2020-07-24 03:54:37.353 NONE  [386981] [OPropagator::resize@218]  m_oevt 0x20ad0310 evt 0x576b580 numPhotons 11235 u_numPhotons 11235
    2020-07-24 03:54:37.353 NONE  [386981] [OPropagator::setSize@152]  width 11235 height 1
    2020-07-24 03:54:37.353 NONE  [386981] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-24 03:54:37.353 INFO  [386981] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 15 0 0) -
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 202 identity (     202       5       9       1 ) offset      202  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 203 identity (     203       4      10       1 ) offset      203  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       10        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       10        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       10        0 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 202 identity (     202       5       9       1 ) offset      202  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0        9        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0        9        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0        9        0 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 identity (     200       5       9       1 ) offset      200  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 identity (     201       4      10       1 ) offset      201  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       21        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       21        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       21        0 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 identity (     200       5       9       1 ) offset      200  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 identity (     201       4      10       1 ) offset      201  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       15        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       15        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        0 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 identity (     200       5       9       1 ) offset      200  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 identity (     201       4      10       1 ) offset      201  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       15        2) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       15        2) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       15        2 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 identity (     200       5       9       1 ) offset      200  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 identity (     201       4      10       1 ) offset      201  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       22        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       22        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       22        0 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 identity (     199       4      10       1 ) offset      199  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 identity (     200       5       9       1 ) offset      200  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 identity (     201       4      10       1 ) offset      201  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 identity (  142632      35      21   12397 ) offset    21114  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 identity (  142633      30      15   12397 ) offset    21115  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 identity (  142634      34      22   12397 ) offset    21116  
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 identity (  142636      31      26   12397 ) offset    21118  
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (       0        0       26        0) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (       0        0       26        0) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (        0        0       26        0 )
    2020-07-24 03:54:37.636 INFO  [386981] [OPropagator::launch@275] LAUNCH DONE
    2020-07-24 03:54:37.636 INFO  [386981] [OPropagator::launch@277] 0 : (0;11235,1) 
    2020-07-24 03:54:37.636 INFO  [386981] [BTimes::dump@177] OPropagator::launch
                    launch001                 0.283009
    2020-07-24 03:54:37.637 INFO  [386981] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-07-24 03:54:37.681 INFO  [386981] [NPho::Dump@141] OEvent::downloadHitsCompute --dumphit,post,flgs ox Y
    2020-07-24 03:54:37.682 INFO  [386981] [NPho::dump@179] NPho::Dump desc NPho 5804,4,4 numPhotons 5804 maxDump 100 numDump 100
     i       0 post (  13298.711  4694.703-13193.136      70.456) flgs     -27       0       4    2114
     i       1 post (  -7995.062 13221.808-11531.257      70.829) flgs     -27       0       5    2114
     i       2 post ( -11036.258  8525.500-13451.145      76.670) flgs     -25       0       6    2114
     i       3 post ( -18908.377  2247.879 -3200.770      70.476) flgs     -25       0       8    2114




Arghh FOUND IT : trivial bug : BOOLEAN_DEBUG WAS PARTIALLY STOMPING ON instanceIdentity
------------------------------------------------------------------------------------------


::

    2020-07-24 04:30:29.439 NONE  [443481] [OPropagator::setSize@152]  width 11235 height 1
    2020-07-24 04:30:29.439 NONE  [443481] [OPropagator::launch@250]  _prelaunch 1 m_width 11235 m_height 1
    2020-07-24 04:30:29.439 INFO  [443481] [OPropagator::launch@266] LAUNCH NOW  --printenabled  printLaunchIndex ( 15 0 0) -
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 202 instanceIdentity (     202       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 203 instanceIdentity (     203       4      10       1 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (     203        4       10        1) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (     203        4       10        1) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (     203        4       10        1) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (      203        4       10        1 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 202 instanceIdentity (     202       5       9       1 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (     202        5        9        1) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (     202        5        9        1) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (     202        5        9        1) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (      202        5        9        1 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 instanceIdentity (     200       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 instanceIdentity (     201       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 instanceIdentity (  142632      35      21   12397 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (  142632       35       21    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (  142632       35       21    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (  142632       35       21    12397) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (   142632       35       21    12397 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 instanceIdentity (     200       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 instanceIdentity (     201       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 instanceIdentity (  142632      35      21   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 instanceIdentity (  142633      30      15   12397 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (  142633       30       15    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (  142633       30       15    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (  142633       30       15    12397) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (   142633       30       15    12397 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 instanceIdentity (     200       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 instanceIdentity (     201       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 instanceIdentity (  142632      35      21   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 instanceIdentity (  142633      30      15   12397 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (  142633       30       15    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (  142633       30       15    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (  142633       30       15    12397) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (   142633       30       15    12397 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 instanceIdentity (     200       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 instanceIdentity (     201       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 instanceIdentity (  142632      35      21   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 instanceIdentity (  142633      30      15   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 instanceIdentity (  142634      34      22   12397 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (  142634       34       22    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (  142634       34       22    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (  142634       34       22    12397) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (   142634       34       22    12397 )
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 199 instanceIdentity (     199       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 200 instanceIdentity (     200       5       9       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 0 instance_index 0 primitive_count   0 primIdx 201 instanceIdentity (     201       4      10       1 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   0 instanceIdentity (  142632      35      21   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   1 instanceIdentity (  142633      30      15   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   2 instanceIdentity (  142634      34      22   12397 )   
    // csg_intersect_boolean.h:evaluative_csg WITH_PRINT_IDENTITY_INTERSECT_TAIL repeat_index 3 instance_index 3519 primitive_count   6 primIdx   4 instanceIdentity (  142636      31      26   12397 )   
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH instanceIdentity (  142636       31       26    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity (  142636       31       26    12397) 
    // material1_propagate.cu WITH_PRINT_IDENTITY_CH prd.identity_xyzw (  142636       31       26    12397) 
    //generate.cu WITH_PRINT_IDENTITY prd.identity (   142636       31       26    12397 )
    2020-07-24 04:30:29.729 INFO  [443481] [OPropagator::launch@275] LAUNCH DONE
    2020-07-24 04:30:29.729 INFO  [443481] [OPropagator::launch@277] 0 : (0;11235,1) 
    2020-07-24 04:30:29.729 INFO  [443481] [BTimes::dump@177] OPropagator::launch
                    launch001                 0.289463
    2020-07-24 04:30:29.729 INFO  [443481] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-07-24 04:30:29.778 INFO  [443481] [NPho::Dump@141] OEvent::downloadHitsCompute --dumphit,post,flgs ox Y
    2020-07-24 04:30:29.779 INFO  [443481] [NPho::dump@179] NPho::Dump desc NPho 5804,4,4 numPhotons 5804 maxDump 100 numDump 100
     i       0 post (  13298.711  4694.703-13193.136      70.456) flgs     -27   14975       4    2114
     i       1 post (  -7995.062 13221.808-11531.257      70.829) flgs     -27   14207       5    2114
     i       2 post ( -11036.258  8525.500-13451.145      76.670) flgs     -25   15029       6    2114
     i       3 post ( -18908.377  2247.879 -3200.770      70.476) flgs     -25   10497       8    2114
     i       4 post (   8192.179  2356.565 17389.451      74.135) flgs     -27     762       9    2114
     i       5 post (  -6467.069 -9432.470-15642.414      77.069) flgs     -27   15923      11    2114
     i       6 post (  10826.938  3761.312 15558.440      75.163) flgs     -25    1650      14    2114



