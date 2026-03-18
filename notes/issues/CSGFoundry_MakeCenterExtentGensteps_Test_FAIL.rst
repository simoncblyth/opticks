CSGFoundry_MakeCenterExtentGensteps_Test
==========================================


SEvt::addGenstep sim assert
----------------------------

::

    23/43 Test #23: CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test ......***Failed    3.24 sec
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh - using external config for GEOM J25_7_2_opticks_Debug J25_7_2_opticks_Debug_CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/CSG/tests
                    GEOM : J25_7_2_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh
              EXECUTABLE : CSGFoundry_MakeCenterExtentGensteps_Test
                    ARGS : 
    2026-03-18 20:24:51.903 INFO  [4083914] [SEventConfig::SetDevice@1843] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 257079968
    HeuristicMaxSlot(VRAM)/M         : 257
    HeuristicMaxSlot_Rounded(VRAM)   : 257000000
    MaxSlot/M                        : 0
    ModeLite                         : 0
    ModeMerge                        : 0

    2026-03-18 20:24:51.903 INFO  [4083914] [SEventConfig::SetDevice@1858]  Configured_MaxSlot/M 0 Final_MaxSlot/M 257 HeuristicMaxSlot_Rounded/M 257 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2026-03-18 20:24:53.046 INFO  [4083914] [SFrameGenstep::Maybe_Add_PRIOR_SIMTRACE_Genstep@414] with_PRIOR_SIMTRACE NO  prim 4 uprim 4 PRIOR_SIMTRACE_PRIM -2 PRIOR_SIMTRACE - gs_PRIOR_SIMTRACE -
    CSGFoundry_MakeCenterExtentGensteps_Test: /home/blyth/opticks/sysrap/SEvt.cc:2471: sgs SEvt::addGenstep(const quad6&): Assertion `sim' failed.
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh: line 58: 4083914 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Debug/bin/CSGTestRunner.sh : FAIL from CSGFoundry_MakeCenterExtentGensteps_Test



Reproduce
--------------

::

    ok) A[blyth@localhost CSG]$ ~/o/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.sh dbg
    ...
    Reading symbols from CSGFoundry_MakeCenterExtentGensteps_Test...
    Starting program: /data1/blyth/local/opticks_Debug/lib/CSGFoundry_MakeCenterExtentGensteps_Test 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [New Thread 0x7ffff01ff000 (LWP 4084373)]
    2026-03-18 20:28:18.596 INFO  [4084370] [SEventConfig::SetDevice@1843] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 257079968
    HeuristicMaxSlot(VRAM)/M         : 257
    HeuristicMaxSlot_Rounded(VRAM)   : 257000000
    MaxSlot/M                        : 0
    ModeLite                         : 0
    ModeMerge                        : 0

    2026-03-18 20:28:18.597 INFO  [4084370] [SEventConfig::SetDevice@1858]  Configured_MaxSlot/M 0 Final_MaxSlot/M 257 HeuristicMaxSlot_Rounded/M 257 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    [Detaching after vfork from child process 4084378]
    2026-03-18 20:28:19.760 INFO  [4084370] [SFrameGenstep::Maybe_Add_PRIOR_SIMTRACE_Genstep@414] with_PRIOR_SIMTRACE NO  prim 0 uprim 0 PRIOR_SIMTRACE_PRIM -2 PRIOR_SIMTRACE - gs_PRIOR_SIMTRACE -
    CSGFoundry_MakeCenterExtentGensteps_Test: /home/blyth/opticks/sysrap/SEvt.cc:2471: sgs SEvt::addGenstep(const quad6&): Assertion `sim' failed.

    Thread 1 "CSGFoundry_Make" received signal SIGABRT, Aborted.
    0x00007ffff608bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff608bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff603eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff6028833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff602875b in __assert_fail_base.cold () from /lib64/libc.so.6
    #4  0x00007ffff6037886 in __assert_fail () from /lib64/libc.so.6
    #5  0x00007ffff6f6dfb8 in SEvt::addGenstep (this=0x4bc410, q_=...) at /home/blyth/opticks/sysrap/SEvt.cc:2471
    #6  0x00007ffff6f6dbbc in SEvt::addGenstep (this=0x4bc410, a=0x1318a400) at /home/blyth/opticks/sysrap/SEvt.cc:2407
    #7  0x00007ffff6f6a9c9 in SEvt::AddGenstep (a=0x1318a400) at /home/blyth/opticks/sysrap/SEvt.cc:1574
    #8  0x000000000040a024 in main (argc=1, argv=0x7fffffffb418) at /home/blyth/opticks/CSG/tests/CSGFoundry_MakeCenterExtentGensteps_Test.cc:40
    (gdb) 



