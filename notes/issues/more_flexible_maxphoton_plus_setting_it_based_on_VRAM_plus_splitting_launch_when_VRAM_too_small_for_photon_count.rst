more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================

integrate + test new functionality
------------------------------------

* TODO : test bash level use of the new functionality qudarap-prepare-installation 

* DONE : first impl of OPTICKS_MAX_PHOTON:0 to use Heuristic max photon based on VRAM, see SEventConfig::SetDevice


more_flexible_maxphoton
-------------------------

* work on this in::

     ~/o/sysrap/SCurandState.h 
     ~/o/sysrap/tests/SCurandState_test.sh  


* PREVIOUSLY : the maxphoton values that can be used depend on the SCurandState files that have been generated
  and those files are very repetitive and large 

* DONE : use chunk files and concatenate the appropriate number for the 
  desired maxphoton, avoiding duplication and adding flexibility

* DONE : also do partial reads on the last chunk to decouple file sizes from maxphoton

* DONE : comparisions at M3, M10 level between old and new using QRngTest.sh match perfectly 

* DONE : M100 matches after avoid arrays more that 2GB by using TEST=generate only and reducing NV from 16 to 4

* DONE : find and fix the source of the 2GB truncation : NP.hh 

* DONE : 200M matches in TEST=generate with NV=4 



Relevant sources
-------------------

::

    P[blyth@localhost opticks]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   notes/issues/more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count.rst
        modified:   qudarap/QCurandState.cc
        modified:   qudarap/QCurandState.hh
        modified:   qudarap/QRng.cc
        modified:   qudarap/tests/QCurandStateTest.cc
        modified:   sysrap/SCurandState.cc
        modified:   sysrap/SCurandState.h
        modified:   sysrap/sdirectory.h
        modified:   sysrap/tests/SCurandState_test.cc
        modified:   sysrap/tests/SCurandState_test.sh
        modified:   sysrap/tests/SLaunchSequence_test.cc
        modified:   sysrap/tests/sdirectory_test.cc

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        sysrap/tests/SLaunchSequence_test.sh
        sysrap/tests/sdirectory_test.sh


FIXED reversion 
-----------------

::


    FAILS:  6   / 215   :  Mon Dec  2 19:38:33 2024   
      2  /21  Test #2  : QUDARapTest.QRngTest                          ***Failed                      0.18   
      6  /21  Test #6  : QUDARapTest.QSimTest                          ***Failed                      4.46   
      12 /21  Test #12 : QUDARapTest.QSim_Lifecycle_Test               ***Failed                      4.41   
      13 /21  Test #13 : QUDARapTest.QSimWithEventTest                 ***Failed                      4.26   
      21 /21  Test #21 : QUDARapTest.QCurandStateTest                  ***Failed                      0.28   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test         ***Failed                      33.06  



M10 matches
----------------

Generation runs both tests with ALL, analsis checks need to specify one TEST:: 

    ## switch QRng.hh to OLD_MONOLITHIC_CURANDSTATE and recompile
    ## do this first as have to pick the num to match the monolithic file

    TEST=ALL OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh info_run


    ## switch QRng.hh to NEW_CHUNKED_CURANDSTATE and recompile

    TEST=ALL OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh info_run


    PICK=AB TEST=generate      OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh pdb 
    PICK=AB TEST=generate_evid OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh pdb 
  
 


::

    P[blyth@localhost qudarap]$ PICK=AB TEST=generate      OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh pdb 
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py:TEST:generate PICK:AB FOLD:/data/blyth/opticks/QRngTest reldir:None
    -rw-rw-r--. 1 blyth blyth 640000128 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/uu.npy
    -rw-rw-r--. 1 blyth blyth 640000128 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/uu.npy
    au.shape
     (10000000, 16)
    bu.shape
     (10000000, 16)
    au_bu_match:1


    P[blyth@localhost qudarap]$ PICK=AB TEST=generate_evid  OPTICKS_MAX_PHOTON=M10 ~/o/qudarap/tests/QRngTest.sh pdb 
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py:TEST:generate_evid PICK:AB FOLD:/data/blyth/opticks/QRngTest reldir:None
    -rw-rw-r--. 1 blyth blyth 640000128 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/uu.npy
    -rw-rw-r--. 1 blyth blyth 640000128 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/uu.npy
    auu.shape
     (3, 10000000, 16)
    buu.shape
     (3, 10000000, 16)
    auu_buu_match:1



 
M100 SEGV (OOM?) with TEST=ALL try just TEST=generate
--------------------------------------------------------------

::

    P[blyth@localhost qudarap]$ TEST=generate OPTICKS_MAX_PHOTON=M100 ~/o/qudarap/tests/QRngTest.sh run


    P[blyth@localhost qudarap]$ TEST=generate OPTICKS_MAX_PHOTON=M100 ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest
    QRng::LoadAndUpload complete YES rngmax/M 100 rngmax 100000000 digest 499fd4401da334627b6be5ea24d90f19
    2024-12-02 21:27:45.014 INFO  [138428] [QRngTest::QRngTest@42] QRng::desc path /home/blyth/.opticks/rngcache/RNG rngmax 100000000 rngmax/M 100 qr 0x1a75c30 qr.skipahead_event_offset 1 d_qr 0x7f077a200000QRng::Desc IMPL:CHUNKED_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 100000000 nv 16 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest rc:0
    P[blyth@localhost qudarap]$ 

Digest of the chunked read of 100M curandState from 10 chunk files matches the Monolithic file md5sum::

    P[blyth@localhost RNG]$ md5sum QCurandStateMonolithic_100M_0_0.bin
    499fd4401da334627b6be5ea24d90f19  QCurandStateMonolithic_100M_0_0.bin
    P[blyth@localhost RNG]$


::

    P[blyth@localhost qudarap]$ TEST=generate OPTICKS_MAX_PHOTON=M100 ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest
    2024-12-02 21:32:14.710 INFO  [148077] [QRngTest::QRngTest@42] QRng::desc path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_100M_0_0.bin rngmax 100000000 rngmax/M 100 qr 0x179d740 qr.skipahead_event_offset 1 d_qr 0x7f3a64200000QRng::Desc IMPL:OLD_MONOLITHIC_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 100000000 nv 16 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest rc:0
    P[blyth@localhost qudarap]$ 


::

    PICK=AB TEST=generate ~/o/qudarap/tests/QRngTest.sh pdb


Looks like truncation of array to 2GB somewhere::

    P[blyth@localhost qudarap]$ PICK=AB TEST=generate ~/o/qudarap/tests/QRngTest.sh pdb
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py:TEST:generate PICK:AB FOLD:/data/blyth/opticks/QRngTest reldir:None
    -rw-rw-r--. 1 blyth blyth 2105032832 Dec  2 21:28 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/u_0.npy
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py in <module>
        135 
        136     elif PICK == "AB":
    --> 137         a = QRngTest(a_reldir)
        138         b = QRngTest(b_reldir)
        139 

    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py in __init__(self, reldir)
         29         if os.path.exists(upath):
         30             os.system("ls -l %s" % upath)
    ---> 31             u = np.load(upath)
         32         pass
         33         if os.path.exists(uupath):
    ...
    ValueError: cannot reshape array of size 526258176 into shape (100000000,16)
    > /home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py(771)read_array()



Reproduce that error in  ~/np/tests/NP_Make_test.sh
------------------------------------------------------

* ~/o/notes/issues/NP_Make_2GB_truncation_int_bytes_somewhere.rst




::

    P[blyth@localhost RNG]$ cd /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/
    P[blyth@localhost CHUNKED_CURANDSTATE]$ l
    total 4028392
    2055700 -rw-rw-r--. 1 blyth blyth 2105032832 Dec  2 21:28 u_0.npy
    1875004 -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:15 uu.npy
          0 drwxr-xr-x. 4 blyth blyth         67 Dec  2 16:08 ..
         28 -rw-rw-r--. 1 blyth blyth      27216 Dec  2 15:51 fig.png
          0 drwxr-xr-x. 2 blyth blyth         63 Dec  2 15:50 .
      97660 -rw-rw-r--. 1 blyth blyth  100000128 Dec  2 15:06 u.npy
    P[blyth@localhost CHUNKED_CURANDSTATE]$ du -h u_0.npy
    2.0G    u_0.npy
    P[blyth@localhost CHUNKED_CURANDSTATE]$ cd ..
    P[blyth@localhost float]$ l
    total 0
    0 drwxr-xr-x. 2 blyth blyth 35 Dec  2 16:30 OLD_MONOLITHIC_CURANDSTATE
    0 drwxr-xr-x. 4 blyth blyth 67 Dec  2 16:08 .
    0 drwxr-xr-x. 2 blyth blyth 63 Dec  2 15:50 CHUNKED_CURANDSTATE
    0 drwxr-xr-x. 3 blyth blyth 19 Dec  2 14:23 ..
    P[blyth@localhost float]$ cd OLD_MONOLITHIC_CURANDSTATE/
    P[blyth@localhost OLD_MONOLITHIC_CURANDSTATE]$ l
    total 3930704
    2055700 -rw-rw-r--. 1 blyth blyth 2105032832 Dec  2 21:32 u_0.npy
    1875004 -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:12 uu.npy
          0 drwxr-xr-x. 2 blyth blyth         35 Dec  2 16:30 .
          0 drwxr-xr-x. 4 blyth blyth         67 Dec  2 16:08 ..
    P[blyth@localhost OLD_MONOLITHIC_CURANDSTATE]$ du -h u_0.npy
    2.0G    u_0.npy
    P[blyth@localhost OLD_MONOLITHIC_CURANDSTATE]$ 




FIXED : Cause of 2 GB truncation ? Maybe largest int limitation somewhere ? NP.hh ?
---------------------------------------------------------------------------------------------

::

    In [4]: 1024*1024*1024*2
    Out[4]: 2147483648

    In [5]: 0x1 << 31
    Out[5]: 2147483648



Reduce NV from 16 to 4 : reduces file size to 1.5G : then M100 generate test matches
---------------------------------------------------------------------------------------

::

    P[blyth@localhost float]$ du -h */u_0.npy
    1.5G    CHUNKED_CURANDSTATE/u_0.npy
    1.5G    OLD_MONOLITHIC_CURANDSTATE/u_0.npy



::

    P[blyth@localhost tests]$ TEST=generate OPTICKS_MAX_PHOTON=M100 ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest
    2024-12-02 21:45:21.075 INFO  [169928] [QRngTest::QRngTest@42] QRng::desc path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_100M_0_0.bin rngmax 100000000 rngmax/M 100 qr 0xc96780 qr.skipahead_event_offset 1 d_qr 0x7fa514200000QRng::Desc IMPL:OLD_MONOLITHIC_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 100000000 nv 4 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest rc:0
    P[blyth@localhost tests]$ 

    P[blyth@localhost tests]$ TEST=generate OPTICKS_MAX_PHOTON=M100 ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest
    QRng::LoadAndUpload complete YES rngmax/M 100 rngmax 100000000 digest 499fd4401da334627b6be5ea24d90f19
    2024-12-02 21:47:40.733 INFO  [175078] [QRngTest::QRngTest@42] QRng::desc path /home/blyth/.opticks/rngcache/RNG rngmax 100000000 rngmax/M 100 qr 0x2cf6cc0 qr.skipahead_event_offset 1 d_qr 0x7f957a200000QRng::Desc IMPL:CHUNKED_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 100000000 nv 4 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest rc:0
    P[blyth@localhost tests]$ 


    P[blyth@localhost tests]$ PICK=AB TEST=generate ~/o/qudarap/tests/QRngTest.sh pdb
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py:TEST:generate PICK:AB FOLD:/data/blyth/opticks/QRngTest reldir:None
    -rw-rw-r--. 1 blyth blyth 1600000128 Dec  2 21:47 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/uu.npy
    -rw-rw-r--. 1 blyth blyth 1600000128 Dec  2 21:45 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/uu.npy
    au.shape
     (100000000, 4)
    bu.shape
     (100000000, 4)
    au_bu_match:1



M200 QRngTest.sh : matched
----------------------------

::

    OPTICKS_MAX_PHOTON=M200 QRngTest__generate_NV=4 TEST=generate ~/o/qudarap/tests/QRngTest.sh run

    P[blyth@localhost qudarap]$ OPTICKS_MAX_PHOTON=M200 QRngTest__generate_NV=4 TEST=generate ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest QRng::IMPL[CHUNKED_CURANDSTATE]
    QRng::LoadAndUpload complete YES rngmax/M 200 rngmax 200000000 digest 6a2d46957f64e6e1bc459c538a503a58
    2024-12-04 16:51:51.124 INFO  [315082] [QRngTest::QRngTest@44] QRng::desc path /home/blyth/.opticks/rngcache/RNG rngmax 200000000 rngmax/M 200 qr 0x1c56290 qr.skipahead_event_offset 1 d_qr 0x7f0672400000QRng::Desc IMPL:CHUNKED_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 200000000 nv 4 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest QRng::IMPL[CHUNKED_CURANDSTATE] rc:0
    P[blyth@localhost qudarap]$ 


Chunked read digest matches the monolithic file::

    P[blyth@localhost RNG]$ md5sum QCurandStateMonolithic_200M_0_0.bin 
    6a2d46957f64e6e1bc459c538a503a58  QCurandStateMonolithic_200M_0_0.bin
    P[blyth@localhost RNG]$ 

    P[blyth@localhost CHUNKED_CURANDSTATE]$ du -h u_0.npy
    3.0G    u_0.npy
    P[blyth@localhost CHUNKED_CURANDSTATE]$ ls -l u_0.npy
    -rw-rw-r--. 1 blyth blyth 3200000128 Dec  4 16:52 u_0.npy

Expected filesize in bytes with the 128 byte header::

    In [3]: 200*1000000*4*4   
    Out[3]: 3200000000


Flip the switch in QRng.hh::

    qu
    vi QRng.hh
    om


Run again::

    P[blyth@localhost qudarap]$ OPTICKS_MAX_PHOTON=M200 QRngTest__generate_NV=4 TEST=generate ~/o/qudarap/tests/QRngTest.sh run
    [main argv[0] QRngTest QRng::IMPL[OLD_MONOLITHIC_CURANDSTATE]
    2024-12-04 17:00:42.702 INFO  [332488] [QRngTest::QRngTest@44] QRng::desc path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_200M_0_0.bin rngmax 200000000 rngmax/M 200 qr 0x1c25780 qr.skipahead_event_offset 1 d_qr 0x7f1cd4400000QRng::Desc IMPL:OLD_MONOLITHIC_CURANDSTATE
    [QRngTest::main TEST:[generate]
    //QRng_generate ni 200000000 nv 4 skipahead 0 
    ]QRngTest::main rc:0
    ]main argv[0] QRngTest QRng::IMPL[OLD_MONOLITHIC_CURANDSTATE] rc:0
    P[blyth@localhost qudarap]$ 


Compare, matches::

    P[blyth@localhost qudarap]$ PICK=AB TEST=generate ~/o/qudarap/tests/QRngTest.sh pdb
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    /data/blyth/junotop/opticks/qudarap/tests/QRngTest.py:TEST:generate PICK:AB FOLD:/data/blyth/opticks/QRngTest reldir:None
    -rw-rw-r--. 1 blyth blyth 3200000128 Dec  4 16:52 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:15 /data/blyth/opticks/QRngTest/float/CHUNKED_CURANDSTATE/uu.npy
    -rw-rw-r--. 1 blyth blyth 3200000128 Dec  4 17:00 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/u_0.npy
    -rw-rw-r--. 1 blyth blyth 1920000144 Dec  2 21:12 /data/blyth/opticks/QRngTest/float/OLD_MONOLITHIC_CURANDSTATE/uu.npy
    au.shape
     (200000000, 4)
    bu.shape
     (200000000, 4)
    au_bu_match:1


    In [1]: au
    Out[1]: 
    array([[0.74022, 0.43845, 0.51701, 0.15699],
           [0.92099, 0.46036, 0.33346, 0.37252],
           [0.03902, 0.25021, 0.18448, 0.96242],
           [0.96896, 0.49474, 0.67338, 0.56277],
           ...,
           [0.32596, 0.06075, 0.70001, 0.15792],
           [0.77092, 0.73217, 0.99293, 0.66166],
           [0.07743, 0.88589, 0.13311, 0.08525],
           [0.14177, 0.65988, 0.77002, 0.99305]], dtype=float32)

    In [2]: bu
    Out[2]: 
    array([[0.74022, 0.43845, 0.51701, 0.15699],
           [0.92099, 0.46036, 0.33346, 0.37252],
           [0.03902, 0.25021, 0.18448, 0.96242],
           [0.96896, 0.49474, 0.67338, 0.56277],
           ...,
           [0.32596, 0.06075, 0.70001, 0.15792],
           [0.77092, 0.73217, 0.99293, 0.66166],
           [0.07743, 0.88589, 0.13311, 0.08525],
           [0.14177, 0.65988, 0.77002, 0.99305]], dtype=float32)


    In [3]: np.where(au == 0.)
    Out[3]: (array([], dtype=int64), array([], dtype=int64))

    In [4]: np.where(bu == 0.)
    Out[4]: (array([], dtype=int64), array([], dtype=int64))

    In [5]: np.where(bu == 1.)
    Out[5]: 
    (array([ 45494023,  56700706,  87388694, 106515917, 109731375, 115817628, 120388692, 128290047, 132224065, 140951702, 145019702, 147138470, 164947865, 166091746, 167762821, 168649102, 170550692,
            176719626, 179621639, 195222672, 195762010, 196989351]),
     array([0, 1, 1, 3, 1, 0, 2, 1, 1, 1, 2, 0, 1, 0, 1, 3, 0, 0, 0, 0, 3, 2]))

    In [6]: np.where(au == 1.)
    Out[6]: 
    (array([ 45494023,  56700706,  87388694, 106515917, 109731375, 115817628, 120388692, 128290047, 132224065, 140951702, 145019702, 147138470, 164947865, 166091746, 167762821, 168649102, 170550692,
            176719626, 179621639, 195222672, 195762010, 196989351]),
     array([0, 1, 1, 3, 1, 0, 2, 1, 1, 1, 2, 0, 1, 0, 1, 3, 0, 0, 0, 0, 3, 2]))

    In [7]: 





VRAM detection
-----------------

Do that at initialization just before loading states, sdevice is already in use somewhere, 
mainly for metadata purposes. Maybe will need to move it earlier for this purpose. 

* cuda has device API : ~/o/sysrap/sdevice.h  uses that 
* nvml has C api : ~/o/sysrap/smonitor.{sh,cc} uses that 


::

    P[blyth@localhost qudarap]$ opticks-f sdevice.h
    ./sysrap/CMakeLists.txt:    sdevice.h
    ./sysrap/scontext.h:scontext.h : holds sdevice.h structs for all and visible GPUs
    ./sysrap/scontext.h:    ./sysrap/sdevice.h
    ./sysrap/scontext.h:#include "sdevice.h"
    ./sysrap/sdevice.h:sdevice.h 
    ./sysrap/sdevice.h:and metadata recording is handled with sdevice.h scontext.h 
    ./sysrap/sdevice.h:* scontext.h needs updating to handle updated sdevice.h and 
    ./sysrap/tests/sdevice_test.cc:#include "sdevice.h"
    ./sysrap/tests/sdevice_test.sh:into run/event metadata. Or could access the sdevice.h struct 

    P[blyth@localhost opticks]$ opticks-f scontext.h
    ./CSGOptiX/CSGOptiX.cc:#include "scontext.h"   // GPU metadata
    ./sysrap/CMakeLists.txt:    scontext.h
    ./sysrap/scontext.h:scontext.h : holds sdevice.h structs for all and visible GPUs
    ./sysrap/scontext.h:    [blyth@localhost sysrap]$ opticks-fl scontext.h 
    ./sysrap/scontext.h:    ./sysrap/scontext.h
    ./sysrap/sdevice.h:and metadata recording is handled with sdevice.h scontext.h 
    ./sysrap/sdevice.h:* scontext.h needs updating to handle updated sdevice.h and 
    ./sysrap/tests/scontext_test.cc:#include "scontext.h"



Currently scontext lives up in cx::

     293 /**
     294 CSGOptiX::InitMeta
     295 -------------------
     296 
     297 **/
     298 
     299 void CSGOptiX::InitMeta(const SSim* ssim  )
     300 {
     301     std::string gm = GetGPUMeta() ;            // (QSim) scontext sdevice::brief
     302     SEvt::SetRunMetaString("GPUMeta", gm.c_str() );  // set CUDA_VISIBLE_DEVICES to control 
     303 


     386 scontext* CSGOptiX::SCTX = nullptr ;
     387 
     388 
     389 /**
     390 CSGOptiX::SetSCTX
     391 ---------------------
     392 
     393 Instanciates CSGOptiX::SCTX(scontext) holding GPU metadata. 
     394 Canonically invoked from head of CSGOptiX::Create.
     395 
     396 NOTE: Have sometimes observed few second hangs checking for GPU 
     397 
     398 **/
     399 
     400 void CSGOptiX::SetSCTX()
     401 {
     402     LOG(LEVEL) << "[ new scontext" ;
     403     SCTX = new scontext ;
     404     LOG(LEVEL) << "] new scontext" ;
     405     LOG(LEVEL) << SCTX->desc() ;
     406 }
     407 
     408 std::string CSGOptiX::GetGPUMeta(){ return SCTX ? SCTX->brief() : "ERR-NO-CSGOptiX-SCTX" ; }
     409 



     344 CSGOptiX* CSGOptiX::Create(CSGFoundry* fd )
     345 {
     346     SProf::Add("CSGOptiX__Create_HEAD");
     347     LOG(LEVEL) << "[ fd.descBase " << ( fd ? fd->descBase() : "-" ) ;
     348 
     349     SetSCTX();
     350     QU::alloc = new salloc ;   // HMM: maybe this belongs better in QSim ? 
     351 
     352     InitEvt(fd);
     353     InitSim( const_cast<SSim*>(fd->sim) ); // QSim instanciation after uploading SSim arrays
     354     InitMeta(fd->sim);                     // recording GPU, switches etc.. into run metadata
     355     InitGeo(fd);                           // uploads geometry 
     356 
     357     CSGOptiX* cx = new CSGOptiX(fd) ;



But the config is down at SEventConfig level::

     237 int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
     238 int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
     239 int SEventConfig::MaxSimtrace(){   return _MaxSimtrace ; }
     240 int SEventConfig::MaxCurandState(){ return std::max( MaxPhoton(), MaxSimtrace() ) ; }


Move scontext booting down to SEventConfig::Initialize ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nope the natural place to instanciate scontext is SEventConfig::Initialize  
in order to control the scontext/SEventConfig interaction 
and use VRAM results in the config. 


HMM signaling with OPTICKS_MAX_PHOTON=0 is problematic
---------------------------------------------------------------

SEventConfig.cc::

     940     else if(IsRGModeSimulate())
     941     {
     942         gather_mask |= SCOMP_DOMAIN ;  save_mask |= SCOMP_DOMAIN ;
     943 
     944         if(MaxGenstep()>0){  gather_mask |= SCOMP_GENSTEP ; save_mask |= SCOMP_GENSTEP ; }
     945         if(MaxPhoton()>0)
     946         {
     947             gather_mask |= SCOMP_INPHOTON ;  save_mask |= SCOMP_INPHOTON ;
     948             gather_mask |= SCOMP_PHOTON   ;  save_mask |= SCOMP_PHOTON   ;
     949             gather_mask |= SCOMP_HIT      ;  save_mask |= SCOMP_HIT ;
     950             //gather_mask |= SCOMP_SEED ;   save_mask |= SCOMP_SEED ;  // only needed for deep debugging 
     951         }


Perhaps use MaxCurandState : because that is more explicit, regarding defining the Max Launch Slots ?

::

    P[blyth@localhost opticks]$ opticks-f MaxCurandState
    ./qudarap/QRng.cc:    rngmax(SEventConfig::MaxCurandState()),    // max of : OPTICKS_MAX_PHOTON OPTICKS_MAX_SIMTRACE 
    ./sysrap/SEventConfig.cc:int SEventConfig::MaxCurandState(){ return std::max( MaxPhoton(), MaxSimtrace() ) ; }
    ./sysrap/SEventConfig.cc:       << std::setw(20) << " MaxCurandState " << " : " << MaxCurandState() 
    ./sysrap/SEventConfig.cc:       << std::setw(20) << " MaxCurandState/M " << " : " << MaxCurandState()/M
    ./sysrap/SEventConfig.cc:    meta->set_meta<int>("MaxCurandState", MaxCurandState() );  
    ./sysrap/SEventConfig.hh:MaxCurandState
    ./sysrap/SEventConfig.hh:    static int MaxCurandState();  // from max of MaxPhoton and MaxSimtrace
    ./sysrap/SCurandStateMonolithic.cc:       << " SEventConfig::MaxCurandState() " << SEventConfig::MaxCurandState() << std::endl
    ./sysrap/SCurandStateMonolithic.cc:        int rngmax = SEventConfig::MaxCurandState(); 
    P[blyth@localhost opticks]$ 


::

     38 /**
     39 QRng::QRng
     40 ------------
     41 
     42 QRng instanciation is invoked from QSim::UploadComponents
     43 
     44 **/
     45 
     46 QRng::QRng(unsigned skipahead_event_offset)
     47     :
     48 #ifdef OLD_MONOLITHIC_CURANDSTATE
     49     path(SCurandStateMonolithic::Path()),        // null path will assert in Load
     50     rngmax(0),
     51     d_rng_states(LoadAndUpload(rngmax, path)),   // rngmax set based on file_size/item_size of path 
     52 #else
     53     cs(nullptr),
     54     path(cs.getDir()),                        // informational 
     55     rngmax(SEventConfig::MaxCurandState()),    // max of : OPTICKS_MAX_PHOTON OPTICKS_MAX_SIMTRACE 
     56     d_rng_states(LoadAndUpload(rngmax, cs)),   // 
     57 #endif
     58     qr(new qrng(d_rng_states, skipahead_event_offset)),
     59     d_qr(nullptr)
     60 {
     61     init();
     62 }






scontext/SEventConfig coordination + booting
------------------------------------------------

::

    P[blyth@localhost sysrap]$ opticks-f SEventConfig.hh
    ./CSG/CSGSimtrace.cc:#include "SEventConfig.hh"
    ./CSG/CSGFoundry.cc:#include "SEventConfig.hh"

          

    ./CSGOptiX/tests/CSGOptiXSimTest.cc:#include "SEventConfig.hh"
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc:#include "SEventConfig.hh"
    ./CSGOptiX/tests/CXRaindropTest.cc:#include "SEventConfig.hh"
    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:#include "SEventConfig.hh"
    ./CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:#include "SEventConfig.hh"
    ./CSGOptiX/CSGOptiX.cc:#include "SEventConfig.hh"
    ./extg4/X4Simtrace.cc:#include "SEventConfig.hh"
    ./g4cx/tests/G4CXSimtraceTest.cc:#include "SEventConfig.hh"
    ./g4cx/tests/G4CXSimulateTest.cc:#include "SEventConfig.hh"
    ./g4cx/tests/G4CXApp.h:#include "SEventConfig.hh"
    ./g4cx/tests/G4CXRenderTest.cc:#include "SEventConfig.hh"
    ./g4cx/G4CXOpticks.cc:#include "SEventConfig.hh"
    ./qudarap/tests/QEvent_Lifecycle_Test.cc:#include "SEventConfig.hh"
    ./qudarap/tests/QSim_Lifecycle_Test.cc:#include "SEventConfig.hh"
    ./qudarap/tests/QSimTest.cc:#include "SEventConfig.hh"

    ./qudarap/QSim.cc:#include "SEventConfig.hh"
    ./qudarap/QEvent.cc:#include "SEventConfig.hh"
    ./qudarap/QRng.cc:#include "SEventConfig.hh"


    ./sysrap/CMakeLists.txt:    SEventConfig.hh
    ./sysrap/SCF.h:#include "SEventConfig.hh"
    ./sysrap/SEventConfig.cc:#include "SEventConfig.hh"
    ./sysrap/SEvt.cc:#include "SEventConfig.hh"
    ./sysrap/SGeo.cc:#include "SEventConfig.hh"
    ./sysrap/SOpticks.cc:#include "SEventConfig.hh"
    ./sysrap/SSimtrace.h:#include "SEventConfig.hh"
    ./sysrap/scontext.h:#include "SEventConfig.hh"
    ./sysrap/tests/SEventConfigTest.cc:#include "SEventConfig.hh"
    ./sysrap/tests/SEvtTest.cc:#include "SEventConfig.hh"
    ./sysrap/sevent.h:#include "SEventConfig.hh"
    ./sysrap/SCurandStateMonolithic.cc:#include "SEventConfig.hh"
    ./u4/U4App.h:#include "SEventConfig.hh"
    ./u4/tests/U4SimtraceTest.cc:#include "SEventConfig.hh"
    ./u4/U4Tree.h:#include "SEventConfig.hh"
    P[blyth@localhost opticks]$ 





WIP: Setting maxphoton based on VRAM
--------------------------------------

Heuristic calculation of maxphoton depends on available VRAM plus the 
array recording that is enabled.  So need to do this in SEventConfig, within

SEventConfig::SetVRAM


NEXT: some big scans with VRAM measurement to improve the Heuristic 


salloc estimate
~~~~~~~~~~~~~~~~~~~

::

     992 uint64_t SEventConfig::EstimateAlloc()
     993 {
     994     salloc* estimate = new salloc ;
     995     uint64_t tot = estimate->get_total() ;
     996     delete estimate ;
     997     return tot ;
     998 }




splitting launch to handle more photon than fit into VRAM
--------------------------------------------------------------

* Easiest to reduce change and do the multiple launches at EndOfEvent


Where/how to split the launch ? QSim::simulate seems best 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What is needed:


1. QSim::simulate collects from SEvt genstep slice structs, each with: 

   * genstep slice indices {start,stop}, eg [0:num_gs] when can do single launch  
   * photon {offset, count} , eg zero offset when can do single launch, count always <= SEventConfig::MaxCurand() 

2. QSim::simulate loops over the genstep slices doing the launches

4. SEvt::gather needs to use the slice struct photon {offset, count} 
   to place the outputs into the correct place in the SEvt arrays at download 
   
   * recall that array saving is a debug activity, so assuming that the arrays 
     fit into CPU memory is allowed 

5. what about hit selection ? more involved because its selection over the photons


   
Note that the below are uploading:: 

    QEvent::setGenstep
    QEvent::setGenstepUpload_NP
    QEvent::setGenstepUpload   



genstep slice generalization 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* uploading sliced genstep can be done within QEvent::setGenstepUpload_NP
  with additional genstep slice struct argument 

::

     211 int QEvent::setGenstepUpload_NP(const NP* gs_)
     212 {
     213     gs = gs_ ;
     214     SGenstep::Check(gs);
     215     LOG(LEVEL)
     216         << " gs " << ( gs ? gs->sstr() : "-" )
     217         << SGenstep::Desc(gs, 10)
     218         ;
     219 
     220     int num_genstep = gs_ ? gs_->shape[0] : 0 ;
     221     const char* data = gs_ ? gs_->bytes() : nullptr ;
     222     const quad6* qq = (const quad6*)data ;
     223     int rc = setGenstepUpload(qq, num_genstep);
     224     return rc ;
     225 }




::

     350 double QSim::simulate(int eventID, bool reset_)
     351 {
     352     SProf::Add("QSim__simulate_HEAD");
     353 
     354     LOG_IF(info, SEvt::LIFECYCLE) << "[ eventID " << eventID ;
     355     if( event == nullptr ) return -1. ;
     356 
     357     sev->beginOfEvent(eventID);  // set SEvt index and tees up frame gensteps for simtrace and input photon simulate running
     358 


     /// need to get slices here 
     /// [ loop over slices 

     359     int rc = event->setGenstep() ;    // QEvent 
     

     360     LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : have event but no gensteps collected : will skip cx.simulate " ;
     361 
     362 
     363     SProf::Add("QSim__simulate_PREL");
     364 
     365     sev->t_PreLaunch = sstamp::Now() ;
     366     double dt = rc == 0 && cx != nullptr ? cx->simulate_launch() : -1. ;  //SCSGOptiX protocol
     367     sev->t_PostLaunch = sstamp::Now() ;
     368     sev->t_Launch = dt ;
     369 
     370     SProf::Add("QSim__simulate_POST");
     371 
     372     sev->gather();

     /// ] end loop over slices 


     373 
     374     SProf::Add("QSim__simulate_DOWN");
     375 
     376     int num_ht = sev->getNumHit() ;   // NB from fold, so requires hits array gathering to be configured to get non-zero 
     377     int num_ph = event->getNumPhoton() ;
     378 
     379     LOG_IF(info, SEvt::MINIMAL)
     380         << " eventID " << eventID
     381         << " dt " << std::setw(11) << std::fixed << std::setprecision(6) << dt
     382         << " ph " << std::setw(10) << num_ph
     383         << " ph/M " << std::setw(10) << num_ph/M
     384         << " ht " << std::setw(10) << num_ht
     385         << " ht/M " << std::setw(10) << num_ht/M
     386         << " reset_ " << ( reset_ ? "YES" : "NO " )
     387         ;
     388 
     389     if(reset_) reset(eventID) ;
     390     SProf::Add("QSim__simulate_TAIL");
     391     return dt ;
     392 }





::

     188 int QEvent::setGenstep()  // onto device
     189 {
     190     LOG_IF(info, SEvt::LIFECYCLE) << "[" ;
     191     
     192     
     193     NP* gs_ = sev->getGenstepArray();
     194     int rc = setGenstepUpload_NP(gs_) ;
     195 
     196     LOG_IF(info, SEvt::LIFECYCLE) << "]" ;
     197 
     198     return rc ;
     199 }





How to gather in slices ?
---------------------------

* (sevent)evt->num_photon adjusted for each sub-launch OR separate slice argument ?
* additional slice arg is cleaner  (or setSlice method)

::

     559 void QEvent::gatherPhoton(NP* p) const
     560 {   
     561     bool expected_shape =  p->has_shape(evt->num_photon, 4, 4) ;  
     562     LOG(expected_shape ? LEVEL : fatal) << "[ evt.num_photon " << evt->num_photon << " p.sstr " << p->sstr() << " evt.photon " << evt->photon ;
     563     assert(expected_shape ); 
     564     int rc = QU::copy_device_to_host<sphoton>( (sphoton*)p->bytes(), evt->photon, evt->num_photon );
     565 
     566     LOG_IF(fatal, rc != 0) 
     567          << " QU::copy_device_to_host photon FAILED "
     568          << " evt->photon " << ( evt->photon ? "Y" : "N" )
     569          << " evt->num_photon " <<  evt->num_photon
     570          ;
     571     
     572     if(rc != 0) std::raise(SIGINT) ;
     573     
     574     LOG(LEVEL) << "] evt.num_photon " << evt->num_photon  ;
     575 }
     576 
     577 NP* QEvent::gatherPhoton() const
     578 {   
     579     //NP* p = NP::Make<float>( evt->num_photon, 4, 4); 
     580     NP* p = sev->makePhoton();
     581     gatherPhoton(p);
     582     return p ;
     583 }




