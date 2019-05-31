ckm-okg4-initial-comparisons
================================

Context :doc:`ckm-okg4-initial-comparisons-reveal-idom-bug`



ckm-okg4 : From gensteps bi-simulation comparison (-1:G4, 1:OK)
--------------------------------------------------------------------

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


OKG4Test events fully instrumented on both sides using the CFG4 CRecorder machinery::

    [blyth@localhost 1]$ np.py tmp/blyth/OKG4Test/evt/g4live/natural/{-1,1} -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1
    . :            tmp/blyth/OKG4Test/evt/g4live/natural/-1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190530-2246 
    . :            tmp/blyth/OKG4Test/evt/g4live/natural/-1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ox.npy :          (221, 4, 4) : 0c933fd9fdab9d2975af9e6871351e46 : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ph.npy :          (221, 1, 2) : 0a50e4992b98714e0391cd6d8deadc9e : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ps.npy :          (221, 1, 4) : 2f17ee76054cc1040f30bee0a8a0153e : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/rs.npy :      (221, 10, 1, 4) : 629500c344dc05dbc6777ccf6f386fe5 : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/rx.npy :      (221, 10, 2, 4) : 2ce8d2aafab81f6d6f0e6a1cc1877646 : 20190530-2246 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190530-2246 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/1
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2246 
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2246 


Observations:

* OK: misses so.npy source photons 

  * TODO: add collection of just generated source photons, behind a WITH_SOURCE_PHOTONS macro, 
    this will be more convenient that having to do a separate bouncemax zero run to check the source photons

* OK: array digests match between 1st executable CerenkovMinimal and 2nd OKG4Test : as is expected because same input gensteps and same code

* G4 : so.npy source photons between 1st and 2nd are not an exact match, but they are very close::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/-1/so.npy tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy
    a :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190530-2247 
    b :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190530-2246 
     max(a-b)   5.96e-08  min(a-b)  -5.96e-08 


* G4 : ht.npy getting zero hits in 2nd executable 



evt.py nload.py needs generalization to find OKG4Test events
-----------------------------------------------------------------








