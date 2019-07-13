revive-production-running-mode
================================

Context
---------

* :doc:`big-running-causing-disk-space-pressure`


isProduction "--production"
------------------------------

::

    [blyth@localhost optickscore]$ opticks-f isProduction 
    ./ok/OKMgr.cc:    bool production = m_ok->isProduction();
     # skips m_hub->anaEvent()

    ./optickscore/Opticks.cc:    ss << ( isProduction() ? " PRODUCTION" : " DEVELOPMENT" ) ;
    ./optickscore/Opticks.cc:bool Opticks::isProduction() const 
    ./optickscore/Opticks.hh:       bool isProduction() const ; // --production

    ./optickscore/OpticksEvent.cc:    if(m_ok->isProduction())
    # skips saving the arrays

    ./optixrap/OEvent.cc:    if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);
    # does not download the evt, only hits    

    ./okop/OpEngine.cc:    if(m_ok->isProduction()) return ; 
    # event indexing is skipped


    ./okop/OpMgr.cc:    bool production = m_ok->isProduction();
    # post saving analysis is skipped


Added "production" to generate.cu, skipping step point recording for production == 0 
---------------------------------------------------------------------------------------------

* this saves nothing but metadata, add "--savehit" 


TODO : some limited downloaded hit analysis, so not flying blind in production running
----------------------------------------------------------------------------------------

::

    n [1]: a
    Out[1]: 
    Evt( 30,"torch","tboolean-box",pfx="tboolean-box", seqs="[]", msli="0:100k:" ) 20190713-2037 
    /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/30
     file_photons -   load_slice -   loaded_photons - 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:100000000  
         gs :            - :              : (gensteps) 
         ox :            - :              : (photons) final photon step MISSING  
         ht :            - :    15675,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :        15675 : (hits) wavelength 
      hpost :            - :      15675,4 : (hits) final photon step: position, time 
      hdirw :            - :      15675,4 : (hits) final photon step: direction, weight  
      hpolw :            - :      15675,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :        15675 : (hits) final photon step: flags  
        hc4 :            - :        15675 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx :            - :              : (records) photon step records 
         ph :            - :              : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 

    In [2]: a.ht
    Out[2]: 
    A(torch,30,tboolean-box)-
    A([[[  58.5492,  170.2793,  450.    ,    3.8635],
        [   0.4733,    0.1947,    0.8591,    1.    ],
        [  -0.6335,    0.7529,    0.1783,  380.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],

       [[-191.6261, -240.3635,  450.    ,    5.366 ],
        [  -0.6321,   -0.301 ,    0.714 ,    1.    ],
        [  -0.1712,    0.9529,    0.2502,  380.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],

       [[ 185.7709, -133.8455,  450.    ,    7.1141],
        [   0.645 ,   -0.1747,    0.7439,    1.    ],
        [   0.1911,    0.9795,    0.0644,  380.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],




::

    OpticksProfile=ERROR TBOOLEAN_TAG=40  ts box --generateoverride -40  --rngmax 100  --nog4propagate --compute --cvd 1 --rtx 1 --production --savehit    ## 40M, 0.2383

    ta box --tag 40 




    In [1]: a
    Out[1]: 
    Evt( 40,"torch","tboolean-box",pfx="tboolean-box", seqs="[]", msli="0:100k:" ) 20190713-2047 
    /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/40
     file_photons -   load_slice -   loaded_photons - 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:100000000  
         gs :            - :              : (gensteps) 
         ox :            - :              : (photons) final photon step MISSING  
         ht :            - :    20802,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :        20802 : (hits) wavelength 
      hpost :            - :      20802,4 : (hits) final photon step: position, time 
      hdirw :            - :      20802,4 : (hits) final photon step: direction, weight  
      hpolw :            - :      20802,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :        20802 : (hits) final photon step: flags  
        hc4 :            - :        20802 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx :            - :              : (records) photon step records 
         ph :            - :              : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 

    In [2]: b
    Out[2]: 
    Evt(-40,"torch","tboolean-box",pfx="tboolean-box", seqs="[]", msli="0:100k:" ) 20190713-2047 
    /tmp/blyth/opticks/tboolean-box/evt/tboolean-box/torch/-40
     file_photons -   load_slice -   loaded_photons - 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:100000000  
         gs :            - :              : (gensteps) 
         ox :            - :              : (photons) final photon step MISSING  
         ht :            - :        0,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :            0 : (hits) wavelength 
      hpost :            - :          0,4 : (hits) final photon step: position, time 
      hdirw :            - :          0,4 : (hits) final photon step: direction, weight  
      hpolw :            - :          0,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :            0 : (hits) final photon step: flags  
        hc4 :            - :            0 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx :            - :              : (records) photon step records 
         ph :            - :              : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 



Even with "--nog4propagate" are getting an empty ht.npy file for G4 side ?::

    [blyth@localhost -40]$ np.py  ht.npy -v
    a :                                                       ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190713-2047 
    (0, 4, 4)
    f32
    []


::

    1586 /**
    1587 OpticksEvent::saveHitData
    1588 --------------------------
    1589 
    1590 Writes hit buffer even when empty, otherwise get inconsistent 
    1591 buffer time stamps when changes makes hits go away and are writing 
    1592 into the same directory.
    1593 
    1594 Argument form allows externals like G4Opticks to save Geant4 sourced
    1595 hit data collected with CPhotonCollector into an event dir 
    1596 with minimal fuss. 
    1597 
    1598 **/
    1599 

