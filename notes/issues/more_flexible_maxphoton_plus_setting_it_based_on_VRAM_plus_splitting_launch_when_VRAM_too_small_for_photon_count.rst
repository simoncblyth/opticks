more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================

integrate + test new functionality
------------------------------------

* TODO : test bash level use of the new functionality qudarap-prepare-installation 

* DONE : first impl of OPTICKS_MAX_PHOTON:0 to use Heuristic max photon based on VRAM, see SEventConfig::SetDevice


OPTICKS_MAX_PHOTON/OPTICKS_MAX_CURAND/OPTICKS_MAX_SLOT as what is VRAM constrained ? 
--------------------------------------------------------------------------------------

OPTICKS_MAX_PHOTON 
   no longer makes sense as the thing that is constrained, as are removing the constraint 
   by splitting launches

OPTICKS_MAX_CURAND
   also does not make sense as the constrained thing if provide reproducibility 
   despite split launches by uploading appropriate ranges of curandState 

   * number of curandState will need to exceed launch slots 

OPTICKS_MAX_SLOT
   current preference : explicit that it is a technical maximum coming from available VRAM 


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

* YEP: moved to SEventConfig::Initialize_Meta

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



Old scontext was invoked up in cx::

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


DONE : Move scontext booting down to SEventConfig::Initialize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The natural place to instanciate scontext is SEventConfig::Initialize  
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
   * photon {offset, count} , eg zero offset when can do single launch, count always <= SEventConfig::MaxSlot() 
   * DONE : SGenstep::GetGenstepSlices 


2. QSim::simulate loops over the genstep slices doing the launches

4. SEvt::gather needs to use the slice struct photon {offset, count} 
   to place the outputs into the correct place in the SEvt arrays at download 
   
   * recall that array saving is a debug activity, so assuming that the arrays 
     fit into CPU memory is allowed 

5. what about hit selection ? more involved because its selection over the photons


   


QSim/QEvent how to use the genstep slices ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

::

    QEvent::setGenstep
    QEvent::setGenstepUpload_NP
    QEvent::setGenstepUpload   


genstep slice generalization 

* uploading sliced genstep can be done within QEvent::setGenstepUpload_NP
  with additional genstep slice struct argument 

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



how to gather outputs and place them at appropriate offset positions in arrays 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* initial CPU array creation before the loop which 
  gets offset populated by each launch ?

* thats very different to current approach of creating the arrays 
  for each component and adding them to the fold : instead will 
  have to create the fold with empty arrays and then offset populate them. 
  Thats doable for most components where the array sizes are known ahead of time.  
  But cannot be done like that for hits : the most important one. 

* for hits do not know the size will get from each launch, 
  so will just need to gather multiple NP arrays and concatenate them.

* actually the way SEvt::gather_components is implemented it is easier to collect
  (NPFold)fold for each launch and concatenate the arrays from within those fold 

* actually could do this for all components, array collection other than hits
  is a debugging activity : so should not expend effort to make it fast/memory-efficient

* ordinarily would not have arrays (other than hit) enabled when doing 
  large simulations that need split launches  


::

    3484 void SEvt::gather_components()   // *GATHER*
    3485 {
    3486     int num_genstep = -1 ;
    3487     int num_photon  = -1 ;
    3488     int num_hit     = -1 ;
    3489 
    3490     int num_comp = gather_comp.size() ;
    3491 
    3492     LOG(LEVEL) << " num_comp " << num_comp << " from provider " << provider->getTypeName() ;
    3493     LOG_IF(info, GATHER) << " num_comp " << num_comp << " from provider " << provider->getTypeName() ;
    3494 
    3495 
    3496     for(int i=0 ; i < num_comp ; i++)
    3497     {
    3498         unsigned cmp = gather_comp[i] ;
    3499         const char* k = SComp::Name(cmp);
    3500         NP* a = provider->gatherComponent(cmp);
    3501         bool null_component = a == nullptr ;
    3502 
    3503         LOG(LEVEL)
    3504             << " k " << std::setw(15) << k
    3505             << " a " << ( a ? a->brief() : "-" )
    3506             << " null_component " << ( null_component ? "YES" : "NO " )
    3507             ;
    3508 
    3509         LOG_IF(info, GATHER)
    3510             << " k " << std::setw(15) << k
    3511             << " a " << ( a ? a->brief() : "-" )
    3512             << " null_component " << ( null_component ? "YES" : "NO " )
    3513             ;
    3514 
    3515 
    3516 
    3517 
    3518         if(null_component) continue ;
    3519         fold->add(k, a);
    3520 
    3521         int num = a->shape[0] ;
    3522         if(     SComp::IsGenstep(cmp)) num_genstep = num ;
    3523         else if(SComp::IsPhoton(cmp))  num_photon = num ;
    3524         else if(SComp::IsHit(cmp))     num_hit = num ;
    3525     }
    3526 
    3527     gather_total += 1 ;
    3528 
    3529     if(num_genstep > -1) genstep_total += num_genstep ;
    3530     if(num_photon > -1)  photon_total += num_photon ;
    3531     if(num_hit > -1)     hit_total += num_hit ;



::

     572 void QEvent::gatherPhoton(NP* p) const
     573 {
     574     bool expected_shape =  p->has_shape(evt->num_photon, 4, 4) ;
     575     LOG(expected_shape ? LEVEL : fatal) << "[ evt.num_photon " << evt->num_photon << " p.sstr " << p->sstr() << " evt.photon " << evt->photon ;
     576     assert(expected_shape );
     577     int rc = QU::copy_device_to_host<sphoton>( (sphoton*)p->bytes(), evt->photon, evt->num_photon );
     578 
     579     LOG_IF(fatal, rc != 0)
     580          << " QU::copy_device_to_host photon FAILED "
     581          << " evt->photon " << ( evt->photon ? "Y" : "N" )
     582          << " evt->num_photon " <<  evt->num_photon
     583          ;
     584 
     585     if(rc != 0) std::raise(SIGINT) ;
     586 
     587     LOG(LEVEL) << "] evt.num_photon " << evt->num_photon  ;
     588 }
     589 
     590 NP* QEvent::gatherPhoton() const
     591 {
     592     //NP* p = NP::Make<float>( evt->num_photon, 4, 4); 
     593     NP* p = sev->makePhoton();
     594     gatherPhoton(p);
     595     return p ;
     596 }



Modify NPFold::add OR create NPFold{Collection/Set/Seq} OR use two level NPFold/NPFold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* NPFold::add to collect multiple arrays under the same key ?
* OR: come up with keys hit_0 hit_1 ... for each launch 
* OR: keep NPFold asis but create NPFoldCollection that manages multiple NPFold with associated index 
  and does the concatenation in a way hidden from user : so the NPFoldCollection presents
  an API just like NPFold but with the addition of an index 
* HMM: NPFold can already contain other NPFold : so can just use two level NPFold/NPFold, 
  with some added ConcatIfNeeded methods that do nothing if the NPFold layout only one level 

* SEvt could hold "topfold/efold/evtfold" plus "slicefold/currentfold/subfold/fold" 
  pointing to the current within topfold

  * "topfold" and "subfold" would be the same for non multi-launch  
  * for two level NPFold with keys that are repeated across sibling NPFold 
    the NPFold::Concat would concat the subfold into topfold then delete the 2nd level subfold

* Need NPFold::getDepth NPFold::getMaxTreeDepth


SEvt (NPFold)fold lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q: where to use topfold and where fold ? 
Q: where to "fold = topfold->add_subfold()"  ? 
Q: where to "topfold->concat()" ?


* NPFold(fold) instanciated with SEvt  (hmm maybe impl NPFoldCollection) 


::

    0159 SEvt::SEvt()
     160     :
     161     cfgrc(SEventConfig::Initialize()),
     162     index(MISSING_INDEX),
     163     instance(MISSING_INSTANCE),

     192     provider(this),   // overridden with SEvt::setCompProvider for device running from QEvent::init 
     193     fold(new NPFold),


     257 void SEvt::setFoldVerbose(bool v)
     258 {
     259     fold->set_verbose(v);
     260 }
     261 
     262 
     263 const char* SEvt::GetSaveDir(int idx) // static 
     264 {
     265     return Exists(idx) ? Get(idx)->getSaveDir() : nullptr ;
     266 }
     267 const char* SEvt::getSaveDir() const { return fold->savedir ; }
     268 const char* SEvt::getLoadDir() const { return fold->loaddir ; }
     269 int SEvt::getTotalItems() const { return fold->total_items() ; }
     270 

     632 const NP* SEvt::getG4State() const {  return fold->get(SComp::Name(SCOMP_G4STATE)) ; }


    1810 void SEvt::clear_output()
    1811 {   
    1812     setStage(SEvt__clear_output);
    1813 
    1814     LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_output_vector " ;
    1815 
    1816     clear_output_vector();
    1817 
    1818     const char* keylist = "genstep" ;
    1819     bool copy = false ;
    1820     char delim = ',' ;
    1821 
    1822     fold->clear_except(keylist, copy, delim );
    1823 
    1824     LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_output_vector " ;
    1825 
    1826     LOG(LEVEL) << "]" ;
    1827 }


Input gensteps would be directly into topfold::

    1829 void SEvt::clear_genstep()
    1830 {
    1831     setStage(SEvt__clear_genstep);
    1832     LOG_IF(info, LIFECYCLE) << id() << " BEFORE clear_genstep_vector " ; 
    1833 
    1834     clear_genstep_vector();
    1835     fold->clear_only("genstep", false, ',');
    1836 
    1837     LOG_IF(info, LIFECYCLE) << id() << " AFTER clear_genstep_vector " ;
    1838 }

Output arrays collected into 2nd level fold::

    3487 void SEvt::gather_components()   // *GATHER*
    3488 {
    3489     int num_genstep = -1 ;
    3490     int num_photon  = -1 ;
    3491     int num_hit     = -1 ;
    3492 
    3493     int num_comp = gather_comp.size() ;
    3494 
    3495     LOG(LEVEL) << " num_comp " << num_comp << " from provider " << provider->getTypeName() ;
    3496     LOG_IF(info, GATHER) << " num_comp " << num_comp << " from provider " << provider->getTypeName() ;
    3497 
    3498 
    3499     for(int i=0 ; i < num_comp ; i++)
    3500     {
    3501         unsigned cmp = gather_comp[i] ;
    3502         const char* k = SComp::Name(cmp);
    3503         NP* a = provider->gatherComponent(cmp);  // see QEvent::gatherComponent for GPU running 
    3504         bool null_component = a == nullptr ;
    3505 
    3506         LOG(LEVEL)
    3507             << " k " << std::setw(15) << k
    3508             << " a " << ( a ? a->brief() : "-" )
    3509             << " null_component " << ( null_component ? "YES" : "NO " )
    3510             ;
    3511 
    3512         LOG_IF(info, GATHER)
    3513             << " k " << std::setw(15) << k
    3514             << " a " << ( a ? a->brief() : "-" )
    3515             << " null_component " << ( null_component ? "YES" : "NO " )
    3516             ;
    3517 
    3518 
    3519 
    3520 
    3521         if(null_component) continue ;
    3522         fold->add(k, a);
    3523 
    3524         int num = a->shape[0] ;
    3525         if(     SComp::IsGenstep(cmp)) num_genstep = num ;
    3526         else if(SComp::IsPhoton(cmp))  num_photon = num ;
    3527         else if(SComp::IsHit(cmp))     num_hit = num ;
    3528     }
    3529 


The subfold would normally be transient in memory only, 
so things like metadata not specific to the launch would need 
to go into topfold::

    3596 void SEvt::add_array( const char* k, const NP* a )
    3597 {
    3598     LOG(LEVEL) << " k " << k << " a " << ( a ? a->sstr() : "-" ) ;
    3599     fold->add(k, a);
    3600 }
    3601 
    3602 void SEvt::addEventConfigArray()
    3603 {
    3604     fold->add(SEventConfig::NAME, SEventConfig::Serialize() );
    3605 }


The getters would normally be from topfold, unless debugging.
Flexibilitity to switch fold between topfold and the subfold : not worthwhile ?::

   SEvt::setFoldIndex -1:top 0,1,2,3:subfold

Getters:::

    4142 const NP* SEvt::getGenstep() const { return fold->get(SComp::GENSTEP_) ;}
    4143 const NP* SEvt::getPhoton() const {  return fold->get(SComp::PHOTON_) ; }
    4144 const NP* SEvt::getHit() const {     return fold->get(SComp::HIT_) ; }
    4145 const NP* SEvt::getAux() const {     return fold->get(SComp::AUX_) ; }
    4146 const NP* SEvt::getSup() const {     return fold->get(SComp::SUP_) ; }
    4147 const NP* SEvt::getPho() const {     return fold->get(SComp::PHO_) ; }
    4148 const NP* SEvt::getGS() const {      return fold->get(SComp::GS_) ; }
    4149 
    4150 unsigned SEvt::getNumPhoton() const { return fold->get_num(SComp::PHOTON_) ; }
    4151 unsigned SEvt::getNumHit() const
    4152 {
    4153     int num = fold->get_num(SComp::HIT_) ;  // number of items in array 
    4154     return num == NPFold::UNDEF ? 0 : num ;   // avoid returning -1 when no hits
    4155 }



How to change NPFold ?
~~~~~~~~~~~~~~~~~~~~~~~~

* DONE: Added NPFold::concat that concatenates common subfold arrays into top level, 
  so the subfold can correspond to each launch  


control the launch loop
~~~~~~~~~~~~~~~~~~~~~~~~

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




Start adding sliced genstep launch to QEvent SEvt
---------------------------------------------------

::

    FAILS:  3   / 215   :  Sat Dec  7 14:23:59 2024   
      87 /107 Test #87 : SysRapTest.SEvtLoadTest                       ***Exception: SegFault         0.21   
      96 /107 Test #96 : SysRapTest.SEvt_test                          ***Exception: SegFault         0.20    null fold
      10 /21  Test #10 : QUDARapTest.QEventTest                        ***Failed                      0.76    -1 asserts
                                 

    om-test-help
    -------------





How to test sliced launches ? 
---------------------------------

* cxs_min.sh often uses a single torch genstep, need to change that to test genstep sliced multi-launch
* cxs_min.sh large_evt : try to get to 300M 400M with only 24G VRAM 

::

    247 elif [ "$TEST" == "large_evt" ]; then
    248 
    249    opticks_num_photon=M200   ## OOM with TITAN RTX 24G 
    250    opticks_max_photon=M200   ## cost: QRng init time + VRAM 
    251    opticks_num_event=1
    252    opticks_running_mode=SRM_TORCH
    253 


how to config split genstep torch running ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     741 void SEvt::addInputGenstep()
     742 {

     785             else if( has_torch )
     786             {
     787                 if( SEvent::HasGENSTEP() )
     788                 {
     789                     // expected with G4CXApp.h U4Recorder running : see G4CXApp::GeneratePrimaries
     790                     // this is because the gensteps are needed really early with Geant4 running 
     791                     igs = SEvent::GetGENSTEP() ;
     792                 }
     793                 else
     794                 {
     795                     int index_arg = getIndexArg();
     796                     igs = SEvent::MakeTorchGenstep(index_arg);  // pass index to allow changing num photons per event
     797                 }
     798             }
     799             assert(igs); 
     800             addGenstep(igs);



Generalized to fabricate multi-gensteps::

   SEvent::MakeGenstep
   SEventConfig::


   TEST=MakeTorchGenstep ~/o/sysrap/tests/SEventTest.sh


multilaunch test
-----------------

~/o/cxs_min.sh::

    224 elif [ "$TEST" == "ref10_multilaunch" ]; then
    225 
    226    opticks_num_photon=M10
    227    opticks_num_genstep=10
    228    opticks_max_photon=M10
    229    opticks_num_event=1
    230    opticks_running_mode=SRM_TORCH
    231 
    232    export OPTICKS_MAX_SLOT=M1
    233 


::

   LOG=1 ~/o/cxs_min.sh run

   TEST=ref10_multilaunch LOG=1 ~/o/cxs_min.sh run

   TEST=ref10_multilaunch ~/o/cxs_min.sh dbg



10 launches done, but double free at clearing::

    (gdb) bt
    #0  0x00007ffff5854387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5855a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff5896ed7 in __libc_message () from /lib64/libc.so.6
    #3  0x00007ffff589f299 in _int_free () from /lib64/libc.so.6
    #4  0x00007ffff71a76d0 in NP::~NP (this=0x12a44340, __in_chrg=<optimized out>) at /data/blyth/opticks_Debug/include/SysRap/NP.hh:43
    #5  0x00007ffff71c19c1 in NPFold::clear_arrays (this=0x1437a5e0, keep=0x0) at /data/blyth/opticks_Debug/include/SysRap/NPFold.h:1473
    #6  0x00007ffff71c18c3 in NPFold::clear_ (this=0x1437a5e0, keep=0x0) at /data/blyth/opticks_Debug/include/SysRap/NPFold.h:1462
    #7  0x00007ffff71c1891 in NPFold::clear (this=0x1437a5e0) at /data/blyth/opticks_Debug/include/SysRap/NPFold.h:1428
    #8  0x00007ffff71c1ab8 in NPFold::clear_subfold (this=0xeda2d70) at /data/blyth/opticks_Debug/include/SysRap/NPFold.h:1500
    #9  0x00007ffff71b040f in QSim::simulate (this=0x12a41fc0, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:399
    #10 0x00007ffff7c00469 in CSGOptiX::simulate (this=0x12ab38d0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:724
    #11 0x00007ffff7bfd005 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:167
    #12 0x0000000000404a85 in main (argc=1, argv=0x7fffffff42f8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 


    (gdb) f 9
    #9  0x00007ffff71b040f in QSim::simulate (this=0x12a41fc0, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:399
    399     sev->topfold->clear_subfold(); 
    (gdb) list
    394         // trying to use sub fold not top fold
    395 
    396         SProf::Add("QSim__simulate_DOWN"); 
    397     }
    398     sev->topfold->concat(); 
    399     sev->topfold->clear_subfold(); 
    400 
    401     int num_ht = sev->getNumHit() ;   // NB from fold, so requires hits array gathering to be configured to get non-zero 
    402     int num_ph = event->getNumPhoton() ; 
    403 
    (gdb) 


Even with the clear_subfold commented::

     393         sev->gather();
     394         // trying to use sub fold not top fold
     395 
     396         SProf::Add("QSim__simulate_DOWN");
     397     }
     398     sev->topfold->concat();
     399     //sev->topfold->clear_subfold(); 
     400 

get similar at reset::

   TEST=ref10_multilaunch ~/o/cxs_min.sh dbg

    (gdb) bt
    #0  0x00007ffff5854387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5855a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff5896ed7 in __libc_message () from /lib64/libc.so.6
    #3  0x00007ffff589f299 in _int_free () from /lib64/libc.so.6
    #4  0x00007ffff6e85c18 in NP::~NP (this=0x12a44340, __in_chrg=<optimized out>) at /home/blyth/opticks/sysrap/NP.hh:43
    #5  0x00007ffff6f232d9 in NPFold::clear_arrays (this=0x1437a5e0, keep=0x0) at /home/blyth/opticks/sysrap/NPFold.h:1473
    #6  0x00007ffff6f231db in NPFold::clear_ (this=0x1437a5e0, keep=0x0) at /home/blyth/opticks/sysrap/NPFold.h:1462
    #7  0x00007ffff6f231a9 in NPFold::clear (this=0x1437a5e0) at /home/blyth/opticks/sysrap/NPFold.h:1428
    #8  0x00007ffff6f233d0 in NPFold::clear_subfold (this=0xeda2d70) at /home/blyth/opticks/sysrap/NPFold.h:1500
    #9  0x00007ffff6f231e7 in NPFold::clear_ (this=0xeda2d70, keep=0x7fffffff2b40) at /home/blyth/opticks/sysrap/NPFold.h:1463
    #10 0x00007ffff6f235a5 in NPFold::clear_except_ (this=0xeda2d70, keep=std::vector of length 1, capacity 1 = {...}, copy=false) at /home/blyth/opticks/sysrap/NPFold.h:1566
    #11 0x00007ffff6f237da in NPFold::clear_except (this=0xeda2d70, keeplist=0x7ffff703d909 "genstep", copy=false, delim=44 ',') at /home/blyth/opticks/sysrap/NPFold.h:1594
    #12 0x00007ffff6f0726f in SEvt::clear_output (this=0xed59700) at /home/blyth/opticks/sysrap/SEvt.cc:1826
    #13 0x00007ffff6f06460 in SEvt::endOfEvent (this=0xed59700, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1595
    #14 0x00007ffff71b0855 in QSim::reset (this=0x12a41fc0, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:438
    #15 0x00007ffff71b0742 in QSim::simulate (this=0x12a41fc0, eventID=0, reset_=true) at /home/blyth/opticks/qudarap/QSim.cc:414
    #16 0x00007ffff7c00469 in CSGOptiX::simulate (this=0x12ab38d0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:724
    #17 0x00007ffff7bfd005 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:167
    #18 0x0000000000404a85 in main (argc=1, argv=0x7fffffff42f8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 

add NPFold::set_skipdelete_r

export SEvt__NPFOLD_VERBOSE=1


left field
--------------

::

   TEST=ref10_multilaunch ~/o/cxs_min.sh dbg


    2024-12-07 20:51:28.447 INFO  [200267] [QRng::init@72] [QRng__init_VERBOSE] YES
    QRng::desc path /home/blyth/.opticks/rngcache/RNG rngmax 3000000 rngmax/M 3 qr 0x129b1830 qr.skipahead_event_offset 100000 d_qr 0x7fffa4600200QRng::Desc IMPL:CHUNKED_CURANDSTATE
    CSGOptiXSMTest: /home/blyth/opticks/sysrap/NPFold.h:796: void NPFold::add_subfold(const char*, NPFold*): Assertion `fo->parent == nullptr' failed.

    Thread 1 "CSGOptiXSMTest" received signal SIGABRT, Aborted.
    0x00007ffff5850387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5850387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5851a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff58491a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff5849252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff6eb5ae9 in NPFold::add_subfold (this=0x12a1acc0, f=0x7ffff7045255 "jpmt", fo=0xc4ae610) at /home/blyth/opticks/sysrap/NPFold.h:796
    #5  0x00007ffff6f5bbb9 in SPMT::serialize_ (this=0x129ca1b0) at /home/blyth/opticks/sysrap/SPMT.h:584
    #6  0x00007ffff6f5bb5c in SPMT::serialize (this=0x129ca1b0) at /home/blyth/opticks/sysrap/SPMT.h:577
    #7  0x00007ffff6f40753 in SSim::get_spmt_f (this=0x43a440) at /home/blyth/opticks/sysrap/SSim.cc:323
    #8  0x00007ffff71abd0f in QSim::UploadComponents (ssim=0x43a440) at /home/blyth/opticks/qudarap/QSim.cc:183
    #9  0x00007ffff7bfc5f2 in CSGOptiX::InitSim (ssim=0x43a440) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:283
    #10 0x00007ffff7bfcc5b in CSGOptiX::Create (fd=0xec98320) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:352
    #11 0x00007ffff7bfc007 in CSGOptiX::SimulateMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:166
    #12 0x0000000000404a85 in main (argc=1, argv=0x7fffffff42d8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXSMTest.cc:13
    (gdb) 



Tripped because the jpmt comes from SSim top so already has parent::

    0580 inline NPFold* SPMT::serialize_() const   // formerly get_fold 
     581 {
     582     NPFold* fold = new NPFold ;
     583 
     584     if(jpmt) fold->add_subfold("jpmt", const_cast<NPFold*>(jpmt) ) ;
     585 
     586     if(rindex) fold->add("rindex", rindex) ;
     587     if(thickness) fold->add("thickness", thickness) ;
     588     if(qeshape) fold->add("qeshape", qeshape) ;
     589     if(lcqs) fold->add("lcqs", lcqs) ;
     590     return fold ;
     591 }


    310 const NPFold* SSim::get_jpmt() const
    311 {
    312     const NPFold* f = top ? top->find_subfold(JPMT_RELP) : nullptr ;
    313     return f ;
    314 }
    315 const SPMT* SSim::get_spmt() const
    316 {
    317     const NPFold* jpmt = get_jpmt();
    318     return jpmt ? new SPMT(jpmt) : nullptr ;
    319 }
    320 const NPFold* SSim::get_spmt_f() const
    321 {
    322     const SPMT* spmt = get_spmt() ;
    323     const NPFold* spmt_f = spmt ? spmt->serialize() : nullptr ;
    324     return spmt_f ;
    325 }


double free comes from the genstep : because it is repeated into every subfold
-------------------------------------------------------------------------------

In the below the genstep pointers are repeated::

    TEST=ref10_multilaunch ~/o/cxs_min.sh dbg

    Rng::LoadAndUpload complete YES rngmax/M 3 rngmax 3000000 digest c5a80f522e9393efe0302b916affda06
    2024-12-07 21:14:16.495 INFO  [255505] [QRng::init@72] [QRng__init_VERBOSE] YES
    QRng::desc path /home/blyth/.opticks/rngcache/RNG rngmax 3000000 rngmax/M 3 qr 0x129b1830 qr.skipahead_event_offset 100000 d_qr 0x7fffa4600200QRng::Desc IMPL:CHUNKED_CURANDSTATE
    NPFold::add_subfold  WARNING changing parent of added subfold fo 
     fo.treepath [/extra/jpmt]
     fo.parent.treepath [/extra]
     this.treepath []

    NPFold::clear_except( keeplist:genstep copy:0 delim:,)
    NPFold::clear_subfold[]0xeda2d70
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f000/genstep.npy]0x12a44340
    NPFold::add_ [/f000/hit.npy]0x14446570
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f001/genstep.npy]0x12a44340
    NPFold::add_ [/f001/hit.npy]0x14446a20
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f002/genstep.npy]0x12a44340
    NPFold::add_ [/f002/hit.npy]0x1437b360
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f003/genstep.npy]0x12a44340
    NPFold::add_ [/f003/hit.npy]0x1437ba60
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f004/genstep.npy]0x12a44340
    NPFold::add_ [/f004/hit.npy]0x14446c40
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f005/genstep.npy]0x12a44340
    NPFold::add_ [/f005/hit.npy]0x14724890
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f006/genstep.npy]0x12a44340
    NPFold::add_ [/f006/hit.npy]0x147249d0
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f007/genstep.npy]0x12a44340
    NPFold::add_ [/f007/hit.npy]0x147250e0
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f008/genstep.npy]0x12a44340
    NPFold::add_ [/f008/hit.npy]0x14725220
    //qsim::propagate_at_surface_CustomART idx  640276 lpmtid 51681 : ERROR NOT-A-SENSOR : NAN_ABORT 
    NPFold::add_ [/f009/genstep.npy]0x12a44340
    NPFold::add_ [/f009/hit.npy]0x14725360
    NPFold::add_ [/genstep.npy]0x14726220
    NPFold::add_ [/hit.npy]0x147254a0
    NPFold::clear_subfold[]0xeda2d70
    NPFold::clear ALL [/f000]0x14377d50
    NPFold::clear_arrays.delete[/f000/genstep.npy]0x12a44340
    NPFold::clear_arrays.delete[/f000/hit.npy]0x14446570
    NPFold::clear_subfold[/f000]0x14377d50
    NPFold::clear ALL [/f001]0x1437a810
    NPFold::clear_arrays.delete[/f001/genstep.npy]0x12a44340
    *** Error in `/data/blyth/opticks_Debug/lib/CSGOptiXSMTest': double free or corruption (!prev): 0x000000001433e290 ***
    ======= Backtrace: =========
    /lib64/libc.so.6(+0x81299)[0x7ffff589a299]
    /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so(+0x466e0)[0x7ffff71a46e0]
    /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so(+0x6117b)[0x7ffff71bf17b]
    /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so(+0x60fcd)[0x7ffff71befcd]
    /data/blyth/opticks_Debug/lib/../lib64/libQUDARap.so(+0x60f7c)[0x7ffff71bef7c]



The cause is the below "lie" that genstep can be treated like a gathered
output when it is in fact an input::

     898 NP* QEvent::gatherComponent_(unsigned cmp) const
     899 {
     900     NP* a = nullptr ;
     901     switch(cmp)
     902     {
     903         case SCOMP_GENSTEP:   a = getGenstep()     ; break ;
     904         case SCOMP_INPHOTON:  a = getInputPhoton() ; break ;
     905 
     906         case SCOMP_PHOTON:    a = gatherPhoton()   ; break ;
     907         case SCOMP_HIT:       a = gatherHit()      ; break ;
     908 #ifndef PRODUCTION
     909         case SCOMP_DOMAIN:    a = gatherDomain()      ; break ;
     910         case SCOMP_RECORD:    a = gatherRecord()   ; break ;
     911         case SCOMP_REC:       a = gatherRec()      ; break ;
     912         case SCOMP_SEQ:       a = gatherSeq()      ; break ;
     913         case SCOMP_PRD:       a = gatherPrd()      ; break ;



     552 NP* QEvent::getGenstep() const
     553 {
     554     NP* _gs = const_cast<NP*>(gs) ; // const_cast so can use QEvent::gatherComponent_
     555     LOG(LEVEL) << " _gs " << ( _gs ? _gs->sstr() : "-" ) ;
     556     return _gs ;
     557 }
     558 NP* QEvent::getInputPhoton() const
     559 {
     560     return input_photon ;
     561 }



Changing to gatherGenstepFromDevice avoids the double free::

     622 /**
     623 QEvent::gatherGenstepFromDevice
     624 ---------------------------------
     625 
     626 Gensteps originate on host and are uploaded to device, so downloading
     627 them from device is not usually done. It is for debugging only. 
     628 
     629 **/
     630 
     631 NP* QEvent::gatherGenstepFromDevice() const
     632 {
     633     NP* a = NP::Make<float>( evt->num_genstep, 6, 4 );
     634     QU::copy_device_to_host<quad6>( (quad6*)a->bytes(), evt->genstep, evt->num_genstep );
     635     return a ;
     636 }
     637 



Even with KEEP_SUBFOLD no subs are saved::

    2024-12-07 22:08:09.963 INFO  [378837] [QEvent::gatherComponent@886] [ cmp 256 proceed 1 a 0x1474fd70
    NPFold::add_ [/f009/hit.npy]0x1474fd70 (223235, 4, 4, )
    NPFold::add_ [/genstep.npy]0x144187a0 (10, 6, 4, )
    NPFold::add_ [/hit.npy]0x14419fb0 (2232350, 4, 4, )
    2024-12-07 22:08:10.454 INFO  [378837] [QSim::simulate@403]  KEEP_SUBFOLD 
    2024-12-07 22:08:10.454 INFO  [378837] [SEvt::endOfEvent@1590] SEvt::id EGPU (0)  GSV YES SEvt__endOfEvent
    2024-12-07 22:08:10.454 INFO  [378837] [SEvt::save@3967] SEvt::id EGPU (0)  GSV YES SEvt__endOfEvent
    2024-12-07 22:08:10.815 INFO  [378837] [SEvt::clear_output@1821] SEvt::id EGPU (0)  GSV YES SEvt__OTHER BEFORE clear_output_vector 
    NPFold::clear_except( keeplist:genstep copy:0 delim:,)
    NPFold::clear_arrays.delete[/hit.npy]0x14419fb0 (2232350, 4, 4, )
    NPFold::clear_subfold[]0xeda2a80
    NPFold::clear ALL [/f000]0x14377a40
    NPFold::clear_arrays.delete[/f000/genstep.npy]0x14379df0 (1, 6, 4, )
    NPFold::clear_arrays.delete[/f000/hit.npy]0x14446ab0 (223235, 4, 4, )
    NPFold::clear_subfold[/f000]0x14377a40
    NPFold::clear ALL [/f001]0x1437aed0
    NPFold::clear_arrays.delete[/f001/genstep.npy]0x1437b530 (1, 6, 4, )



Thats because NPFold::copy was not including the subfold in SEvt::save::

    3965 void SEvt::save(const char* dir_)
    3966 {
    3967     LOG_IF(info, LIFECYCLE) << id() ;
    3968 
    3969     // former location of the gather 
    3970 
    3971     LOG(LEVEL) << descComponent() ;
    3972     LOG(LEVEL) << descFold() ;
    3973 
    3974     bool shallow = true ; 
    3975     std::string save_comp = SEventConfig::SaveCompLabel() ; 
    3976     NPFold* save_fold = topfold->copy(save_comp.c_str(), shallow) ;
    3977 
    3978     LOG_IF(LEVEL, save_fold == nullptr) << " NOTHING TO SAVE SEventConfig::SaveCompLabel/OPTICKS_SAVE_COMP  " << save_comp ;
    3979     if(save_fold == nullptr) return ;
    3980 
    3981     const NP* seq = save_fold->get("seq");
    3982     NP* seqnib = nullptr ; 
    3983     NP* seqnib_table = nullptr ;
    3984     if(seq)
    3985     {   
    3986         seqnib = CountNibbles(seq) ; 
    3987         seqnib_table = CountNibbles_Table(seqnib) ;  
    3988         save_fold->add("seqnib", seqnib );           
    3989         save_fold->add("seqnib_table", seqnib_table ); 
    3990         // NPFold::add does nothing with nullptr array 
    3991     }
    3992 
    3993 
    3994     int slic = save_fold->_save_local_item_count();
    3995     if( slic > 0 )
    3996     {   
    3997         const char* dir = getOutputDir(dir_);   // THIS CREATES DIRECTORY
    3998         LOG_IF(info, MINIMAL) << dir << " [" << save_comp << "]"  ;
    3999         LOG(LEVEL) << descSaveDir(dir_) ;
    4000 
    4001         LOG(LEVEL) << "[ save_fold.save " << dir ;
    4002         save_fold->save(dir); 
    4003         LOG(LEVEL) << "] save_fold.save " << dir ;
    4004 
    4005         int num_save_comp = SEventConfig::NumSaveComp();
    4006         if(num_save_comp > 0 ) saveFrame(dir);
    4007         // could add frame to the fold ?  
    4008         // for now just restrict to saving frame when other components are saved
    4009     }
    4010     else
    4011     {
    4012         LOG(LEVEL) << "SKIP SAVE AS NPFold::_save_local_item_count zero " ;
    4013     }
    4014 
    4015     // NB: NOT DELETING save_fold AS IT IS A SHALLOW COPY : IT DOES NOT OWN THE ARRAYS 
    4016     delete seqnib ;  
    4017     delete seqnib_table ;  
    4018 }



DONE : add NPFold copying of subfold, for KEEP_SUBFOLD use from SEvt::save, so can check outputs from each slice launch
------------------------------------------------------------------------------------------------------------------------

See::

    ~/np/tests/NPFold_copy_test.sh


Get the subfold::

    TEST=ref10_multilaunch ~/o/cxs_min.sh pdb0

    In [10]: a.f
    Out[10]: 
    a

    CMDLINE:/data/blyth/junotop/opticks/CSGOptiX/cxs_min.py
    a.base:/data/blyth/opticks/GEOM/J_2024nov27/CSGOptiXSMTest/ALL1/A000

      : a.genstep                                          :           (10, 6, 4) : 0:11:56.995033 
      : a.hit                                              :      (2232350, 4, 4) : 0:11:56.971033 
      : a.NPFold_index                                     :                (12,) : 0:11:56.383029 
      : a.NPFold_meta                                      :                   25 : 0:11:56.295029 
      : a.NPFold_names                                     :                 (0,) : 0:11:56.295029 
      : a.sframe                                           :            (4, 4, 4) : 0:11:56.295029 
      : a.sframe_meta                                      :                    5 : 0:11:56.295029 
      : a.f000                                             : SUBFOLD 
      : a.f001                                             : SUBFOLD 
      : a.f002                                             : SUBFOLD 
      : a.f003                                             : SUBFOLD 
      : a.f004                                             : SUBFOLD 
      : a.f005                                             : SUBFOLD 
      : a.f006                                             : SUBFOLD 
      : a.f007                                             : SUBFOLD 
      : a.f008                                             : SUBFOLD 
      : a.f009                                             : SUBFOLD 

     min_stamp : 2024-12-08 14:10:15.887270 
     max_stamp : 2024-12-08 14:10:16.587274 
     dif_stamp : 0:00:00.700004 
     age_stamp : 0:11:56.295029 



FIXED : With subfold trips up sreport
---------------------------------------

* :doc:`sreport_tripped_up_by_keeping_multilaunch_subfold`



TODO : add launch-by-launch curandState uploading OR perhaps load once all chunks and change QRng/qrng slot offsets for each launch
-------------------------------------------------------------------------------------------------------------------------------------


TODO : test exact matching between multi-launch and single launch 
--------------------------------------------------------------------








