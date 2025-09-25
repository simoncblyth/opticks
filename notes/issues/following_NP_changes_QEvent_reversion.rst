following_NP_changes_QEvent_reversion
======================================

Overview
---------

This could be related to the genstep API rejig from a few days ago ?
QEvent::device_alloc_simtrace and other device alloc paths 
trying to alloc num_items=262000000  262M slots   

Actually thats normal. The problem is perhaps that the heuristic 
max slot of 262M is flying close to the actual VRAM ceiling.  
The below worked::

   OPTICKS_MAX_SLOT=M100 ~/o/qudarap/tests/QEventTest.sh
   OPTICKS_MAX_SLOT=M200 ~/o/qudarap/tests/QEventTest.sh
   OPTICKS_MAX_SLOT=M250 ~/o/qudarap/tests/QEventTest.sh
   OPTICKS_MAX_SLOT=M260 ~/o/qudarap/tests/QEventTest.sh
   OPTICKS_MAX_SLOT=M300 ~/o/qudarap/tests/QEventTest.sh

Huhh are exceeding M262 

Also see some variability in what VRAM can reach too, depending
on which other tests were just run. Presumably needs work on cleanup
following tests if these issues continue.  Could also change
the heuristic to be more conservative.

It can also depend on what else is happening on the machine::

    ok) A[blyth@localhost salloc]$ nvidia-smi
    Wed Sep 24 19:41:04 2025       
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA RTX 5000 Ada Gene...    Off |   00000000:AC:00.0  On |                  Off |
    | 30%   38C    P8             15W /  250W |     635MiB /  32760MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A            8416      G   /usr/bin/gnome-shell                    237MiB |
    |    0   N/A  N/A            9179      G   /usr/lib64/firefox/firefox              310MiB |
    |    0   N/A  N/A           10024      G   /usr/bin/Xwayland                        11MiB |
    +-----------------------------------------------------------------------------------------+



Issue
-------

::

    FAILS:  3   / 221   :  Wed Sep 24 17:29:48 2025  :  GEOM J25_4_0_opticks_Debug  
      7  /22  Test #7  : QUDARapTest.QEvent_Lifecycle_Test                       ***Failed                      0.35   
      19 /22  Test #19 : QUDARapTest.QSimWithEventTest                           ***Failed                      3.96   
      21 /22  Test #21 : QUDARapTest.QEventTest                                  ***Failed                      0.33   


::


    (ok) A[blyth@localhost tests]$ ./QEvent_Lifecycle_Test.sh dbg
    ...

    num_event 1000
    2025-09-24 17:32:30.273 ERROR [1663151] [QU::_cudaMalloc@272] save salloc record to /data1/blyth/tmp/GEOM/TEST/QEvent_Lifecycle_Test

    Thread 1 "QEvent_Lifecycl" received signal SIGABRT, Aborted.
    0x00007ffff548bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff548bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff543eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff5428833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff58b135a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff58b13c5 in std::terminate () at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff58b1658 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7ab7800 <typeinfo for QUDA_Exception>, dest=0x7ffff686da7c <QUDA_Exception::~QUDA_Exception()>)
        at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff68f7b33 in QU::_cudaMalloc (p2p=0x7fffffff72b0, size=16768000000, label=0x7ffff69db3b8 "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:287
    #7  0x00007ffff690ae89 in QU::device_alloc_zero<sphoton> (num_items=262000000, label=0x7ffff69db3b8 "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:365
    #8  0x00007ffff68be072 in QEvent::device_alloc_photon (this=0x4ad970) at /home/blyth/opticks/qudarap/QEvent.cc:1070
    #9  0x00007ffff68bdc14 in QEvent::setNumPhoton (this=0x4ad970, num_photon=10000) at /home/blyth/opticks/qudarap/QEvent.cc:1024
    #10 0x00007ffff68b9ad6 in QEvent::setInputPhotonAndUpload (this=0x4ad970) at /home/blyth/opticks/qudarap/QEvent.cc:518
    #11 0x00007ffff68b927d in QEvent::setGenstepUpload (this=0x4ad970, qq0=0xe46aa0, gs_start=0, gs_stop=1) at /home/blyth/opticks/qudarap/QEvent.cc:412
    #12 0x00007ffff68b7dbd in QEvent::setGenstepUpload_NP (this=0x4ad970, gs_=0xe79790, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:220
    #13 0x00007ffff68b7853 in QEvent::setGenstepUpload_NP (this=0x4ad970, gs_=0xe79790) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #14 0x0000000000405b69 in QEvent_Lifecycle_Test::EventLoop () at /home/blyth/opticks/qudarap/tests/QEvent_Lifecycle_Test.cc:61
    #15 0x0000000000405d7e in main (argc=1, argv=0x7fffffffb5a8) at /home/blyth/opticks/qudarap/tests/QEvent_Lifecycle_Test.cc:93
    (gdb) 



    (ok) A[blyth@localhost tests]$ ./QEventTest.sh dbg
    gdb -ex r --args QEventTest
    Wed Sep 24 05:35:08 PM CST 2025
    GNU gdb (AlmaLinux) 14.2-4.1.el9_6
    Copyright (C) 2023 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.
    Type "show copying" and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    Type "show configuration" for configuration details.
    For bug reporting instructions, please see:
    <https://www.gnu.org/software/gdb/bugs/>.
    Find the GDB manual and other documentation resources online at:
        <http://www.gnu.org/software/gdb/documentation/>.

    For help, type "help".
    Type "apropos word" to search for commands related to "word"...
    Reading symbols from QEventTest...
    Starting program: /data1/blyth/local/opticks_Debug/lib/QEventTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [New Thread 0x7fffef1ff000 (LWP 1663362)]
    2025-09-24 17:35:08.760 INFO  [1663359] [SEventConfig::SetDevice@1451] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262326496
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-09-24 17:35:08.760 INFO  [1663359] [SEventConfig::SetDevice@1463]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    QEventTest::setGenstep_one
    [New Thread 0x7fffed54a000 (LWP 1663363)]
    [New Thread 0x7fffecd49000 (LWP 1663364)]
    spath::_ResolvePath token [GEOM] does not resolve 
    2025-09-24 17:35:08.889 ERROR [1663359] [QU::_cudaMalloc@272] save salloc record to /data1/blyth/tmp/GEOM/UNRESOLVED_TOKEN_GEOM/QEventTest

    Thread 1 "QEventTest" received signal SIGABRT, Aborted.
    0x00007ffff508bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff508bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff503eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff5028833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff54b135a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff54b13c5 in std::terminate () at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff54b1658 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7ab7800 <typeinfo for QUDA_Exception>, dest=0x7ffff686da7c <QUDA_Exception::~QUDA_Exception()>)
        at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff68f7b33 in QU::_cudaMalloc (p2p=0x7fffffff6260, size=16768000000, label=0x7ffff69db3b8 "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:287
    #7  0x00007ffff690ae89 in QU::device_alloc_zero<sphoton> (num_items=262000000, label=0x7ffff69db3b8 "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:365
    #8  0x00007ffff68be072 in QEvent::device_alloc_photon (this=0x4a2e50) at /home/blyth/opticks/qudarap/QEvent.cc:1070
    #9  0x00007ffff68bdc14 in QEvent::setNumPhoton (this=0x4a2e50, num_photon=24) at /home/blyth/opticks/qudarap/QEvent.cc:1024
    #10 0x00007ffff68b92bf in QEvent::setGenstepUpload (this=0x4a2e50, qq0=0x4c27f0, gs_start=0, gs_stop=9) at /home/blyth/opticks/qudarap/QEvent.cc:420
    #11 0x00007ffff68b7dbd in QEvent::setGenstepUpload_NP (this=0x4a2e50, gs_=0x48b5d0, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:220
    #12 0x00007ffff68b7853 in QEvent::setGenstepUpload_NP (this=0x4a2e50, gs_=0x48b5d0) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #13 0x0000000000407196 in QEventTest::setGenstep_one () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:94
    #14 0x0000000000409562 in QEventTest::main () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:432
    #15 0x00000000004097a1 in main (argc=1, argv=0x7fffffffb658) at /home/blyth/opticks/qudarap/tests/QEventTest.cc:449
    (gdb) 



QEvent.cc::

     404     int gencode0 = SGenstep::GetGencode(qq, 0) ; // gencode of first genstep or OpticksGenstep_INVALID for qq nullptr
     405 
     406     if(OpticksGenstep_::IsFrame(gencode0))   // OpticksGenstep_FRAME  (HMM: Obtuse, maybe change to SIMTRACE ?)
     407     {
     408         setNumSimtrace( evt->num_seed );
     409     }
     410     else if(OpticksGenstep_::IsInputPhoton(gencode0)) // OpticksGenstep_INPUT_PHOTON  (NOT: _TORCH)
     411     {
     412         setInputPhotonAndUpload();
     413     }
     414     else if(OpticksGenstep_::IsInputPhotonSimtrace(gencode0)) // OpticksGenstep_INPUT_PHOTON_SIMTRACE
     415     {
     416         setInputPhotonSimtraceAndUpload();
     417     }
     418     else
     419     {
     420         setNumPhoton( evt->num_seed );  // *HEAVY* : photon, rec, record may be allocated here depending on SEventConfig
     421     }
     422     upload_count += 1 ;
     423 




TEST=loaded ./QEventTest.sh : too much simtrace ?
----------------------------------------------------

::

    (ok) A[blyth@localhost tests]$ TEST=loaded ./QEventTest.sh dbg
    gdb -ex r --args QEventTest
    Wed Sep 24 07:18:42 PM CST 2025
    GNU gdb (AlmaLinux) 14.2-4.1.el9_6
    ...
    [New Thread 0x7fffef1ff000 (LWP 1669924)]
    2025-09-24 19:18:42.846 INFO  [1669921] [SEventConfig::SetDevice@1451] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33770766336
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262326496
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-09-24 19:18:42.846 INFO  [1669921] [SEventConfig::SetDevice@1463]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    [QEventTest::main ALL NO 
    QEventTest::setGenstep_loaded
    QEventTest::setGenstep_loaded path /data1/blyth/tmp/sysrap/SEventTest/cegs.npy gs0 (49, 6, 4, )
    NP  dtype <f4(49, 6, 4, ) size 1176 uifc f ebyte 4 shape.size 3 data.size 4704 meta.size 0 names.size 0
     array dimensions  ni 49 nj 6 nk 4 item range   i0 0 i1 10 j0 0 j1 6
    [   0]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000   -300.000      1.000
    [   1]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000   -200.000      1.000
    [   2]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000   -100.000      1.000
    [   3]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000      0.000      1.000
    [   4]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000    100.000      1.000
    [   5]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000    200.000      1.000
    [   6]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -300.000      0.000    300.000      1.000
    [   7]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -200.000      0.000   -300.000      1.000
    [   8]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -200.000      0.000   -200.000      1.000
    [   9]  :       0.000      0.000      0.000      0.000 :       0.000      0.000      0.000      1.000 :       1.000      0.000      0.000      0.000 :       0.000      1.000      0.000      0.000 :       0.000      0.000      1.000      0.000 :    -200.000      0.000   -100.000      1.000
    meta:[]
    [QEventTest::setGenstep_loaded gs (49, 6, 4, )
    [New Thread 0x7fffed54a000 (LWP 1669925)]
    [New Thread 0x7fffecd49000 (LWP 1669926)]
    2025-09-24 19:18:42.952 ERROR [1669921] [QU::_cudaMalloc@272] save salloc record to /data1/blyth/tmp/GEOM/DummyGEOMForQEventTest/QEventTest

    Thread 1 "QEventTest" received signal SIGABRT, Aborted.
    0x00007ffff508bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64 nvidia-driver-cuda-libs-580.82.07-1.el9.x86_64
    (gdb) bt
    #0  0x00007ffff508bedc in __pthread_kill_implementation () from /lib64/libc.so.6
    #1  0x00007ffff503eb46 in raise () from /lib64/libc.so.6
    #2  0x00007ffff5028833 in abort () from /lib64/libc.so.6
    #3  0x00007ffff54b135a in __cxxabiv1::__terminate (handler=<optimized out>) at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff54b13c5 in std::terminate () at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff54b1658 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7ab7800 <typeinfo for QUDA_Exception>, dest=0x7ffff686da7c <QUDA_Exception::~QUDA_Exception()>)
        at /opt/conda/conda-bld/gcc-compiler_1654084175708/work/gcc/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff68f7b33 in QU::_cudaMalloc (p2p=0x7fffffff7b20, size=16768000000, label=0x7ffff69db4c8 "QEvent::device_alloc_simtrace/max_slot") at /home/blyth/opticks/qudarap/QU.cc:287
    #7  0x00007ffff6908715 in QU::device_alloc<quad4> (num_items=262000000, label=0x7ffff69db4c8 "QEvent::device_alloc_simtrace/max_slot") at /home/blyth/opticks/qudarap/QU.cc:313
    #8  0x00007ffff68be621 in QEvent::device_alloc_simtrace (this=0x4a2e50) at /home/blyth/opticks/qudarap/QEvent.cc:1107
    #9  0x00007ffff68bdcaa in QEvent::setNumSimtrace (this=0x4a2e50, num_simtrace=490) at /home/blyth/opticks/qudarap/QEvent.cc:1032
    #10 0x00007ffff68b925e in QEvent::setGenstepUpload (this=0x4a2e50, qq0=0x4c4ff0, gs_start=0, gs_stop=49) at /home/blyth/opticks/qudarap/QEvent.cc:408
    #11 0x00007ffff68b7dbd in QEvent::setGenstepUpload_NP (this=0x4a2e50, gs_=0x48b5d0, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:220
    #12 0x00007ffff68b7853 in QEvent::setGenstepUpload_NP (this=0x4a2e50, gs_=0x48b5d0) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #13 0x0000000000408b65 in QEventTest::setGenstep_loaded (gs=0x48b5d0) at /home/blyth/opticks/qudarap/tests/QEventTest.cc:323
    #14 0x0000000000409130 in QEventTest::setGenstep_loaded () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:381
    #15 0x00000000004096e3 in QEventTest::main () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:442
    #16 0x00000000004098dc in main (argc=1, argv=0x7fffffffb638) at /home/blyth/opticks/qudarap/tests/QEventTest.cc:457
    (gdb) 




QEvent::setNumPhoton QEvent::setNumSimtrace
---------------------------------------------

::

    1018 void QEvent::setNumPhoton(unsigned num_photon )
    1019 {
    1020     LOG_IF(info, LIFECYCLE) << " num_photon " << num_photon ;
    1021     LOG(LEVEL);
    1022 
    1023     sev->setNumPhoton(num_photon);
    1024     if( evt->photon == nullptr ) device_alloc_photon();
    1025     uploadEvt();
    1026 }
    1027 
    1028 
    1029 void QEvent::setNumSimtrace(unsigned num_simtrace)
    1030 {
    1031     sev->setNumSimtrace(num_simtrace);
    1032     if( evt->simtrace == nullptr ) device_alloc_simtrace();
    1033     uploadEvt();
    1034 }



    1050 void QEvent::device_alloc_photon()
    1051 {
    1052     LOG_IF(info, LIFECYCLE) ;
    1053     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
    1054 
    1055     LOG(LEVEL)
    1056         << " evt.max_slot   " << evt->max_slot
    1057         << " evt.max_record " << evt->max_record
    1058         << " evt.max_photon " << evt->max_photon
    1059         << " evt.num_photon " << evt->num_photon
    1060 #ifndef PRODUCTION
    1061         << " evt.num_record " << evt->num_record
    1062         << " evt.num_rec    " << evt->num_rec
    1063         << " evt.num_seq    " << evt->num_seq
    1064         << " evt.num_prd    " << evt->num_prd
    1065         << " evt.num_tag    " << evt->num_tag
    1066         << " evt.num_flat   " << evt->num_flat
    1067 #endif
    1068         ;
    1069 
    1070     evt->photon  = evt->max_slot > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot, "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)" ) : nullptr ;
    1071 
    1072 #ifndef PRODUCTION
    1073     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot * evt->max_record, "max_slot*max_record*sizeof(sphoton)" ) : nullptr ;
    1074     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_slot * evt->max_rec   , "max_slot*max_rec*sizeof(srec)"    ) : nullptr ;
    1075     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_slot * evt->max_prd   , "max_slot*max_prd*sizeof(quad2)"    ) : nullptr ;
    1076     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    1077     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_slot                  , "max_slot*sizeof(stag)"    ) : nullptr ;
    1078     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_slot                  , "max_slot*sizeof(sflat)"   ) : nullptr ;
    1079 #endif
    1080 
    1081     LOG(LEVEL) << desc() ;
    1082     LOG(LEVEL) << desc_alloc() ;
    1083 }


    1104 void QEvent::device_alloc_simtrace()
    1105 {
    1106     LOG_IF(info, LIFECYCLE) ;
    1107     evt->simtrace = QU::device_alloc<quad4>( evt->max_slot, "QEvent::device_alloc_simtrace/max_slot" ) ;
    1108     LOG(LEVEL)
    1109         << " evt.num_simtrace " << evt->num_simtrace
    1110         << " evt.max_simtrace " << evt->max_simtrace
    1111         ;
    1112 }



