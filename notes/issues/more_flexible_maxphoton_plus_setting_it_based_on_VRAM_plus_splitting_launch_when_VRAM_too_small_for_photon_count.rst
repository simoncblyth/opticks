more_flexible_maxphoton_plus_setting_it_based_on_VRAM_plus_splitting_launch_when_VRAM_too_small_for_photon_count
==================================================================================================================


more_flexible_maxphoton
-------------------------

* work on this in::

     ~/o/sysrap/SCurandState.h 
     ~/o/sysrap/tests/SCurandState_test.sh  


* PREVIOUSLY : the maxphoton values that can be used depend on the SCurandState files that have been generated
  and those files are very repetitive and large 

* DONE : use chunk files and concatenate the appropriate number for the 
  desired maxphoton, avoiding duplication 

* DONE : also do partial reads on the last chunk to decouple file sizes from maxphoton

* DONE : comparisions at M3, M10 level between old and new using QRngTest.sh match perfectly 

* DONE : M100 matches after avoid arrays more that 2GB by using TEST=generate only and reducing NV from 16 to 4

* TODO : find the source of the 2GB truncation : somewhere using int bytes and hitting the limit ?

* TODO : 200M

* TODO : test bash level use of the new functionality qudarap-prepare-installation 

* TODO : implement OPTICKS_MAX_PHOTON:0 to correspond to maximum permitted by available
         VRAM and modulo the limitation from the available chunks  

         * give warning when the VRAM is large enough to warrant larger launches
           than the chunks permit 


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

    ~/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding)
        438             else:
        439                 return format.read_array(fid, allow_pickle=allow_pickle,
    --> 440                                          pickle_kwargs=pickle_kwargs)
        441         else:
        442             # Try a pickle

    ~/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py in read_array(fp, allow_pickle, pickle_kwargs)
        769             array = array.transpose()
        770         else:
    --> 771             array.shape = shape
        772 
        773     return array

    ValueError: cannot reshape array of size 526258176 into shape (100000000,16)
    > /home/blyth/local/env/tools/conda/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py(771)read_array()
        769             array = array.transpose()
        770         else:
    --> 771             array.shape = shape
        772 
        773     return array

    ipdb>                                                      


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



Cause of 2 GB truncation ? Maybe largest int limitation somewhere ? NP.hh ?
------------------------------------------------------------------------------

::

    In [4]: 1024*1024*1024*2
    Out[4]: 2147483648

    In [5]: 0x1 << 31
    Out[5]: 2147483648



Reduce NV from 16 to 4 : reduces file size to 1.5G : then the generate test matches
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





VRAM detection
-----------------

Do that at initialization just before loading states, sdevice is already in use somewhere, 
mainly for metadata purposes. Maybe will need to move it earlier for this purpose. 

* cuda has device API : ~/o/sysrap/sdevice.h  uses that 
* nvml has C api : ~/o/sysrap/smonitor.{sh,cc} uses that 


Setting maxphoton based on VRAM
--------------------------------



splitting launch to handle more photon that fit into VRAM
--------------------------------------------------------------

* doing all the launches at EndOfEvent ? or doing throughout event ? 
* at the launch boundary : is splitting genstep possible (cloning and changing photon count on the excess) ? 
* how to handle SEvt ?



