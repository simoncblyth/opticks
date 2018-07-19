OKTest --compute --save
===========================

Avoid the interop buffer problem and check the compute only hit saving::

    OKTest --compute --save



Linux/Mac comparison
---------------------

Minimal comparison machinery in bin/ev.bash 


Linux (OptiX_510)
--------------------

* perhaps geometry SD labelling problem (because the mechanics works, as shown by okop-/compactionTest ) 

zero hits::

    2018-07-18 16:59:18.696 INFO  [362349] [OEvent::download@358] OEvent::download id 1
    2018-07-18 16:59:18.731 ERROR [362349] [OEvent::downloadHits@399] OEvent::downloadHits.cpho
    2018-07-18 16:59:18.731 ERROR [362349] [OEvent::downloadHits@401] OEvent::downloadHits.cpho DONE 
    TBuf::TBuf.m_spec : dev_ptr 0x7fc09aa00000 size 100000 num_bytes 6400000 hexdump 0 
    TBuf::TBuf.m_spec : dev_ptr (nil) size 0 num_bytes 0 hexdump 0 
    TBuf::download SKIP  numItems_tbuf 0
    CBufSpec.Summary (empty tbuf?) : dev_ptr (nil) size 0 num_bytes 0 hexdump 0 
    2018-07-18 16:59:18.731 FATAL [362349] [OKPropagator::propagate@84] OKPropagator::propagate(1) DONE nhit: 0
    2018-07-18 16:59:18.731 ERROR [362349] [OpticksEvent::save@1568] skip as CanAnalyse returns false 
    2018-07-18 16:59:18.731 INFO  [362349] [OpticksEvent::save@1579] OpticksEvent::save  id: 1 typ: torch tag: 1 det: dayabay cat:  udet: dayabay num_photons: 100000 num_source : 0 genstep 1,6,4 nopstep 0,4,4 photon 100000,4,4 source NULL record 100000,10,2,4 phosel 100000,1,4 recsel 100000,10,1,4 sequence 100000,1,2 seed 100000,1,1 hit 0,4,4 dir /tmp/blyth/opticks/evt/dayabay/torch/1
    2018-07-18 16:59:18.732 FATAL [362349] [NPY<T>::save@753] NPY values NULL, SKIP attempt to save   itemcount 0 itemshape 4,4 native /tmp/blyth/opticks/evt/dayabay/torch/1/ht.npy
    2018-07-18 16:59:18.732 FATAL [362349] [NPY<T>::save@753] NPY values NULL, SKIP attempt to save   itemcount 0 itemshape 4,4 native /tmp/blyth/opticks/evt/dayabay/torch/1/no.npy
    2018-07-18 16:59:18.732 INFO  [362349] [NPY<T>::dump@1653] OpticksEvent::save (nopstep) (0,4,4) 



macOS (OptiX_501)
--------------------


From logging can see 1060 hits::

    2018-07-18 16:52:28.064 ERROR [1162601] [OEvent::downloadHits@399] OEvent::downloadHits.cpho
    2018-07-18 16:52:28.064 ERROR [1162601] [OEvent::downloadHits@401] OEvent::downloadHits.cpho DONE 
    2018-07-18 16:52:28.068 FATAL [1162601] [OKPropagator::propagate@84] OKPropagator::propagate(1) DONE nhit: 1060
    2018-07-18 16:52:28.068 ERROR [1162601] [OpticksEvent::save@1568] skip as CanAnalyse returns false 
    2018-07-18 16:52:28.068 INFO  [1162601] [OpticksEvent::save@1579] OpticksEvent::save  id: 1 typ: torch tag: 1 det: dayabay cat:  udet: dayabay num_photons: 100000 num_source : 0 genstep 1,6,4 nopstep 0,4,4 photon 100000,4,4 source NULL record 100000,10,2,4 phosel 100000,1,4 recsel 100000,10,1,4 sequence 100000,1,2 seed 100000,1,1 hit 1060,4,4 dir /tmp/blyth/opticks/evt/dayabay/torch/1
    2018-07-18 16:52:28.068 FATAL [1162601] [NPY<float>::save@753] NPY values NULL, SKIP attempt to save   itemcount 0 itemshape 4,4 native /tmp/blyth/opticks/evt/dayabay/torch/1/no.npy
    2018-07-18 16:52:28.068 INFO  [1162601] [NPY<float>::dump@1653] OpticksEvent::save (nopstep) (0,4,4) 


::

    epsilon:okop blyth$ np.py /tmp/blyth/opticks/evt/dayabay/torch/1/
    /tmp/blyth/opticks/evt/dayabay/torch/1
    /tmp/blyth/opticks/evt/dayabay/torch/1/report.txt : 34 
    /tmp/blyth/opticks/evt/dayabay/torch/1/ps.npy : (100000, 1, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/ht.npy : (1060, 4, 4)      ### << 1060 hits 
    /tmp/blyth/opticks/evt/dayabay/torch/1/rx.npy : (100000, 10, 2, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/fdom.npy : (3, 1, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/ox.npy : (100000, 4, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/gs.npy : (1, 6, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/rs.npy : (100000, 10, 1, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/ph.npy : (100000, 1, 2) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/idom.npy : (1, 1, 4) 
    /tmp/blyth/opticks/evt/dayabay/torch/1/20180718_165222/report.txt : 34 

::

    In [18]: h = np.load("/tmp/blyth/opticks/evt/dayabay/torch/1/ht.npy")

    In [19]: h.shape
    Out[19]: (1060, 4, 4)



