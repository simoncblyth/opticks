RESOLVED : precooked_limited_to_100k_photons
================================================

I thought the limit was M1 ? And looks like have 10 files of 100k each.

* Has the auto-concatenation stopped working ?

* No the default was reduced to 100k to avoid slow load presumablu

* Investigated in sysrap/tests/s_seq_test.sh

* added an option to use the 1M precooked::

    export s_seq__SeqPath_DEFAULT_LARGE=1



::

    N[blyth@localhost rng_sequence_f_ni1000000_nj16_nk16_tranche100000]$ du -hs *
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset100000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset200000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset300000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset400000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset500000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset600000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset700000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset800000.npy
    98M	rng_sequence_f_ni100000_nj16_nk16_ioffset900000.npy

    N[blyth@localhost rng_sequence_f_ni1000000_nj16_nk16_tranche100000]$ pwd
    /home/blyth/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000

    N[blyth@localhost rng_sequence_f_ni1000000_nj16_nk16_tranche100000]$ f
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    f

    CMDLINE:/home/blyth/np/f.py
    f.base:.

      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset000000  :     (100000, 16, 16) : 519 days, 22:50:53.320556 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset100000  :     (100000, 16, 16) : 519 days, 22:50:53.245556 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset200000  :     (100000, 16, 16) : 519 days, 22:50:53.171555 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset300000  :     (100000, 16, 16) : 519 days, 22:50:53.096555 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset400000  :     (100000, 16, 16) : 519 days, 22:50:53.023554 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset500000  :     (100000, 16, 16) : 519 days, 22:50:52.949554 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset600000  :     (100000, 16, 16) : 519 days, 22:50:52.875553 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset700000  :     (100000, 16, 16) : 519 days, 22:50:52.802553 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset800000  :     (100000, 16, 16) : 519 days, 22:50:52.727552 
      : f.rng_sequence_f_ni100000_nj16_nk16_ioffset900000  :     (100000, 16, 16) : 519 days, 22:50:52.654552 

     min_stamp : 2022-07-10 20:19:11.612301 
     max_stamp : 2022-07-10 20:19:12.278305 
     dif_stamp : 0:00:00.666004 
     age_stamp : 519 days, 22:50:52.654552 

    In [1]: 





::

    N[blyth@localhost opticks]$ PRECOOKED=1 ./G4CXTest_GEOM.sh 

::

    2023-12-12 18:57:36.953 INFO  [410170] [G4CXApp::BeamOn@342] [ OPTICKS_NUM_EVENT=1
    2023-12-12 18:59:10.215 INFO  [410170] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    SGenerate::GeneratePhotons SGenerate__GeneratePhotons_RNG_PRECOOKED : YES
    s_seq::setSequenceIndexFATAL : OUT OF RANGE :  m_seq_ni 100000 index_ 100000 idx 100000 (must be < m_seq_ni )  desc s_seq::desc
     m_seqpath /home/blyth/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy
     m_seq (100000, 16, 16, )
    G4CXTest: /home/blyth/junotop/ExternalLibs/opticks/head/include/SysRap/s_seq.h:156: void s_seq::setSequenceIndex(int): Assertion `idx_in_range' failed.
    ./G4CXTest_GEOM.sh: line 273: 410170 Aborted                 (core dumped) $bin
    ./G4CXTest_GEOM.sh run error
    N[blyth@localhost opticks]$ 


