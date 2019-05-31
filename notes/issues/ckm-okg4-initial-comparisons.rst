ckm-okg4-initial-comparisons
================================

Context :doc:`ckm-okg4-initial-comparisons-reveal-idom-bug`

Issues
----------

* :doc:`ckm-okg4-initial-comparisons-sensor-matching-yet-again`

  * G4 missing SD/SA : resulting in history divergence 
  * Seen this kinda problem before several times before : sensitivity fails to travel OR be translated  


ckm-- source events
-----------------------

::

    ckm-- () 
    { 
        g4-;
        g4-export;
        CerenkovMinimal
    }



"source" events from uninstrumented on G4 side ckm-- CerenkovMinimal executable, paths relative to geocache dir::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/{-1,1} -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/-1
    . :                          source/evt/g4live/natural/-1/ht.npy :          (108, 4, 4) : f151301a12d1874e9447fd916e7f8719 : 20190530-2247 
    . :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190530-2247 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/1
    . :                         source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2247 
    . :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2247 



ckm-okg4 : From gensteps bi-simulation comparison (-1:G4, 1:OK)
--------------------------------------------------------------------

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }


OKG4Test events fully instrumented on both sides using the CFG4 CRecorder machinery::

    [blyth@localhost 1]$ np.py OKG4Test/evt/g4live/natural/{-1,1} -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/OKG4Test/evt/g4live/natural/-1
    . :                      OKG4Test/evt/g4live/natural/-1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190531-1448 
    . :                      OKG4Test/evt/g4live/natural/-1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ox.npy :          (221, 4, 4) : 0c933fd9fdab9d2975af9e6871351e46 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ph.npy :          (221, 1, 2) : 0a50e4992b98714e0391cd6d8deadc9e : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ps.npy :          (221, 1, 4) : 2f17ee76054cc1040f30bee0a8a0153e : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/rs.npy :      (221, 10, 1, 4) : 629500c344dc05dbc6777ccf6f386fe5 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/rx.npy :      (221, 10, 2, 4) : 2ce8d2aafab81f6d6f0e6a1cc1877646 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1448 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/OKG4Test/evt/g4live/natural/1
    . :                       OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190531-1448 
    . :                       OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190531-1448 


Observations:

* OK: misses so.npy source photons 

  * TODO: add collection of just generated source photons, behind a WITH_SOURCE_PHOTONS macro, 
    this will be more convenient that having to do a separate bouncemax zero run to check the source photons

* OK: array digests match between 1st executable CerenkovMinimal and 2nd OKG4Test : as is expected because same input gensteps and same code

* G4 : so.npy source photons between 1st and 2nd are not an exact match, but they are very close::

    [blyth@localhost 1]$ np.py {source,OKG4Test}/evt/g4live/natural/-1/so.npy 
    a :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190530-2247 
    b :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1448 
     max(a-b)   5.96e-08  min(a-b)  -5.96e-08 


* G4 : ht.npy getting zero hits in 2nd executable 



ckm-ip : after evt.py nload.py pfx generalizations to find OKG4Test events
------------------------------------------------------------------------------

::

    ckm-ip () 
    { 
        ipython --pdb $(which ckm.py) -i $*
    }


Compare history frequency::

    In [1]: a.seqhis_ana.table
    Out[1]: 
    .                seqhis_ana  1:g4live:OKG4Test 
    .                                221         1.00 
    0000              3c1        0.638         141      [3 ] CK BT MI
    0001               71        0.190          42      [2 ] CK SD
    0002               81        0.167          37      [2 ] CK SA
    0003             3cc1        0.005           1      [4 ] CK BT BT MI
    .                                221         1.00 

    In [2]: b.seqhis_ana.table
    Out[2]: 
    .                seqhis_ana  -1:g4live:OKG4Test 
    .                                221         1.00 
    0000              3c1        0.606         134      [3 ] CK BT MI
    0001            3ccc1        0.326          72      [5 ] CK BT BT BT MI
    0002             3cb1        0.023           5      [4 ] CK BR BT MI
    0003           3ccbc1        0.014           3      [6 ] CK BT BR BT BT MI
    0004       bbccbbbbb1        0.009           2      [10] CK BR BR BR BR BR BT BT BR BR
    .                                221         1.00 


* G4 missing SD/SA 


Compare histories of first 20 photons::

    In [6]: a.seqhis_ls[0:20]
    Out[6]: 
    CK SA
    CK SD
    CK BT MI
    CK SA
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK SD
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK SD

    In [7]: b.seqhis_ls[0:20]
    Out[7]: 
    CK BT BT BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT BT BT MI


Recorded positions of first few photons, show they are starting out together but history diverges at the SD/SA which 
happens for 1(OK) but not -1(G4).::

    In [8]: a.rposti(0)
    Out[8]: 
    A()sliced
    A([[  0.061,   0.   ,   0.   ,   0.   ],
       [127.659, -35.981,  89.999,   0.726]])

    In [9]: b.rposti(0)
    Out[9]: 
    A()sliced
    A([[   0.061,    0.   ,    0.   ,    0.   ],
       [ 127.659,  -35.981,   89.999,    0.726],   ### history diverges here, OK ends on an SD, G4 continues to BT on   
       [ 149.876,  -42.268,  109.989,    0.879],
       [ 500.015, -140.996,  356.944,    2.875],
       [ 977.783, -398.114, 1000.   ,    5.683]])

    In [10]: a.rposti(2)
    Out[10]: 
    A()sliced
    A([[   0.336,   -0.061,    0.   ,    0.   ],
       [ 500.015,  206.915, -149.327,    2.576],
       [1000.   ,  521.134, -376.019,    4.682]])

    In [11]: b.rposti(2)
    Out[11]: 
    A()sliced
    A([[   0.336,   -0.061,    0.   ,    0.   ],
       [ 500.015,  206.915, -149.327,    2.576],
       [1000.   ,  521.104, -376.019,    4.682]])



