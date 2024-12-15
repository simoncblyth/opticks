cxs_min_large_evt_shakedown
============================


::

    TEST=large_evt ~/o/cxs_min.sh 


WIP : check large_evt on Ada
--------------------------------------

1. DONE : update /cvmfs with GEOM J_2024nov27
2. DONE: update opticks 
3. DONE : Skip SCurandStateMonolithicTest for RNG_PHILOX
4. WIP: get opticks-t to pass
5. TODO: get ~/o/qudarap/tests/QSimTest_ALL.sh to pass 

::

    A[blyth@localhost ~]$ SCurandStateMonolithicTest
    2024-12-15 22:04:37.604 INFO  [44143] [main@11] 
     spec      1:0:0 num    1000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_1M_0_0.bin exists 1 rngmax 1000000
     spec      3:0:0 num    3000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_3M_0_0.bin exists 0 rngmax 0
     spec     10:0:0 num   10000000 seed          0 offset          0 path /home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_10M_0_0.bin exists 0 rngmax 0
    spath::Filesize unable to open file [/home/blyth/.opticks/rngcache/RNG/QCurandStateMonolithic_3M_0_0.bin]
    SCurandStateMonolithicTest: /home/blyth/opticks/sysrap/spath.h:852: static long int spath::Filesize(const char*): Assertion `!failed' failed.
    Aborted (core dumped)
    A[blyth@localhost ~]$ 


Issue 2 : opticks-t fails from max_photon 1 billion with OOM 
----------------------------------------------------------------

::

    SLOW: tests taking longer that 15 seconds
      108/109 Test #108: SysRapTest.SSimTest                           Passed                         17.82  
      2  /22  Test #2  : QUDARapTest.QRngTest                          ***Failed                      237.74 


    FAILS:  4   / 218   :  Sun Dec 15 22:31:06 2024   
      2  /22  Test #2  : QUDARapTest.QRngTest                          ***Failed                      237.74 
      10 /22  Test #10 : QUDARapTest.QEventTest                        ***Failed                      0.16   
      11 /22  Test #11 : QUDARapTest.QEvent_Lifecycle_Test             ***Failed                      0.18   
      13 /22  Test #13 : QUDARapTest.QSimWithEventTest                 ***Failed                      2.18   


::

    Thread 1 "QEventTest" received signal SIGABRT, Aborted.
    0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5c19a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff635789a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff636336a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff63633d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6363669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75ef8be <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f2f48 in QU::_cudaMalloc (p2p=0x7ffffffed350, size=64000000000, label=0x7ffff76cfc48 "QEvent::device_alloc_photon/max_photon*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:256
    #7  0x00007ffff76067db in QU::device_alloc_zero<sphoton> (num_items=1000000000, label=0x7ffff76cfc48 "QEvent::device_alloc_photon/max_photon*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:332
    #8  0x00007ffff75e9d10 in QEvent::device_alloc_photon (this=0x521ff0) at /home/blyth/opticks/qudarap/QEvent.cc:996
    #9  0x00007ffff75e9916 in QEvent::setNumPhoton (this=0x521ff0, num_photon=24) at /home/blyth/opticks/qudarap/QEvent.cc:952
    #10 0x00007ffff75e5388 in QEvent::setGenstepUpload (this=0x521ff0, qq0=0x521330, gs_start=0, gs_stop=9) at /home/blyth/opticks/qudarap/QEvent.cc:370
    #11 0x00007ffff75e44fb in QEvent::setGenstepUpload_NP (this=0x521ff0, gs_=0x5209a0, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:214
    #12 0x00007ffff75e3f31 in QEvent::setGenstepUpload_NP (this=0x521ff0, gs_=0x5209a0) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #13 0x0000000000409bb1 in QEventTest::setGenstep_one () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:91
    #14 0x000000000040be10 in QEventTest::main () at /home/blyth/opticks/qudarap/tests/QEventTest.cc:425
    #15 0x000000000040c04f in main (argc=1, argv=0x7fffffff0ef8) at /home/blyth/opticks/qudarap/tests/QEventTest.cc:442
    (gdb) f 8
    #8  0x00007ffff75e9d10 in QEvent::device_alloc_photon (this=0x521ff0) at /home/blyth/opticks/qudarap/QEvent.cc:996
    996     evt->photon  = evt->max_photon > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon, "QEvent::device_alloc_photon/max_photon*sizeof(sphoton)" ) : nullptr ; 
    (gdb) p evt->max_photon
    $1 = 1000000000
    (gdb) 


Changing many max_photon to max_slot in QEvent reduces the fails::

    0946 void QEvent::setNumPhoton(unsigned num_photon )
     947 {
     948     LOG_IF(info, LIFECYCLE) << " num_photon " << num_photon ;
     949     LOG(LEVEL);
     950 
     951     sev->setNumPhoton(num_photon);
     952     if( evt->photon == nullptr ) device_alloc_photon();
     953     uploadEvt();
     954 }

     978 void QEvent::device_alloc_photon()
     979 {
     980     LOG_IF(info, LIFECYCLE) ;
     981     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
     982 
     983     LOG(LEVEL)
     984         << " evt.max_photon " << evt->max_photon
     985         << " evt.num_photon " << evt->num_photon
     986 #ifndef PRODUCTION
     987         << " evt.num_record " << evt->num_record
     988         << " evt.num_rec    " << evt->num_rec
     989         << " evt.num_seq    " << evt->num_seq
     990         << " evt.num_prd    " << evt->num_prd
     991         << " evt.num_tag    " << evt->num_tag
     992         << " evt.num_flat   " << evt->num_flat
     993 #endif
     994         ;
     995 
     996     evt->photon  = evt->max_photon > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon, "QEvent::device_alloc_photon/max_photon*sizeof(sphoton)" ) : nullptr ;
     997 
     998 #ifndef PRODUCTION
     999     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_photon * evt->max_record, "max_photon*max_record*sizeof(sphoton)" ) : nullptr ;
    1000     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_photon * evt->max_rec   , "max_photon*max_rec*sizeof(srec)"    ) : nullptr ;
    1001     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_photon * evt->max_prd   , "max_photon*max_prd*sizeof(quad2)"    ) : nullptr ;
    1002     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_photon                  , "max_photon*sizeof(sseq)"    ) : nullptr ;
    1003     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_photon                  , "max_photon*sizeof(stag)"    ) : nullptr ;
    1004     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_photon                  , "max_photon*sizeof(sflat)"   ) : nullptr ;
    1005 #endif
    1006 
    1007     LOG(LEVEL) << desc() ;
    1008     LOG(LEVEL) << desc_alloc() ;
    1009 }



Running QEventTest ALL together in one process fails from OOM. 
Splitting does not::

    P[blyth@localhost tests]$ ./QEventTest_ALL.sh 

    Sun Dec 15 23:14:21 CST 2024

     === 000 === [ TEST=one /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 000 === ] PASS 

     === 001 === [ TEST=sliced /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 001 === ] PASS 

     === 002 === [ TEST=many /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 002 === ] PASS 

     === 003 === [ TEST=loaded /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 003 === ] PASS 

     === 004 === [ TEST=checkEvt /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 004 === ] PASS 

     === 005 === [ TEST=quad6 /data/blyth/junotop/opticks/qudarap/tests/QEventTest.sh 
     === 005 === ] PASS 


    Sun Dec 15 23:14:21 CST 2024
    Sun Dec 15 23:14:28 CST 2024

     TOTAL : 6 
     PASS  : 6 
     FAIL  : 0 




OOM from fake_propagate
----------------------------

::

    P[blyth@localhost tests]$ ./QSimTest_ALL.sh 
    ...

    Sun Dec 15 23:30:19 CST 2024
    Sun Dec 15 23:32:03 CST 2024

     TOTAL : 25 
     PASS  : 24 
     FAIL  : 1 
     === 022 === [ TEST=fake_propagate /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 022 === ] ***FAIL*** 






heuristic max_slot needs to account for enabled arrays, especially record array : get OOM with fake_propagate
-----------------------------------------------------------------------------------------------------------------

::

    TEST=fake_propagate /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh dbg

    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5c19a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff635789a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff636336a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff63633d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6363669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75ef8f0 <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f2f7a in QU::_cudaMalloc (p2p=0x7ffffffef510, size=63040000000, label=0x7ffff76cfc90 "max_slot*max_record*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:256
    #7  0x00007ffff760680d in QU::device_alloc_zero<sphoton> (num_items=985000000, label=0x7ffff76cfc90 "max_slot*max_record*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:332
    #8  0x00007ffff75e9d9f in QEvent::device_alloc_photon (this=0x11809e50) at /home/blyth/opticks/qudarap/QEvent.cc:1000
    #9  0x00007ffff75e9916 in QEvent::setNumPhoton (this=0x11809e50, num_photon=100000) at /home/blyth/opticks/qudarap/QEvent.cc:952
    #10 0x00007ffff75e5a20 in QEvent::setInputPhotonAndUpload (this=0x11809e50) at /home/blyth/opticks/qudarap/QEvent.cc:461
    #11 0x00007ffff75e5365 in QEvent::setGenstepUpload (this=0x11809e50, qq0=0x13cac1d0, gs_start=0, gs_stop=1) at /home/blyth/opticks/qudarap/QEvent.cc:366
    #12 0x00007ffff75e44fb in QEvent::setGenstepUpload_NP (this=0x11809e50, gs_=0x13cac0b0, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:214
    #13 0x00007ffff75e3f31 in QEvent::setGenstepUpload_NP (this=0x11809e50, gs_=0x13cac0b0) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #14 0x00007ffff75b4c14 in QSim::fake_propagate (this=0x11809da0, prd=0x1180bab0, type=38) at /home/blyth/opticks/qudarap/QSim.cc:1189
    #15 0x000000000040caa3 in QSimTest::fake_propagate (this=0x7fffffff3830) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:532
    #16 0x000000000040dc85 in QSimTest::main (this=0x7fffffff3830) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:743
    #17 0x000000000040e1fd in main (argc=1, argv=0x7fffffff3fd8) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:786
    (gdb) 




WIP : review all use of max_photon : many of them need to be max_slot
-----------------------------------------------------------------------------

Setting max_photon to one billion should find issues, via OOM errors. 



FIXED : Issue 1 : genstep slice check rng_state assert
-----------------------------------------------------------

* rngmax messed up by empty string OPTICKS_MAX_CURAND="" leading to max_curand -1::

    In [1]: np.uint64(-1)
    Out[1]: 18446744073709551615

* changed ssys::getenvvar with empty string value to use fallback 
* changed SEventConfig::_MaxCurandDefault to "1" nominal 1 Giga-states 

::

    2024-12-15 19:14:20.432  432273926 : [./cxs_min.sh 
    2024-12-15 19:14:22.218 INFO  [68680] [SEventConfig::SetDevice@1295] SEventConfig::DescDevice
    name                             : NVIDIA TITAN RTX
    totalGlobalMem_bytes             : 25396576256
    totalGlobalMem_GB                : 23
    HeuristicMaxSlot(VRAM)           : 197276976
    HeuristicMaxSlot(VRAM)/M         : 197
    HeuristicMaxSlot_Rounded(VRAM)   : 197000000
    MaxSlot/M                        : 0

    2024-12-15 19:14:22.219 INFO  [68680] [SEventConfig::SetDevice@1307]  Configured_MaxSlot/M 0 Final_MaxSlot/M 197 HeuristicMaxSlot_Rounded/M 197 changed YES
     (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2024-12-15 19:14:22.439 INFO  [68680] [QRng::initStates@72] initStates<Philox> DO NOTHING 
    2024-12-15 19:14:22.439 INFO  [68680] [QRng::init@100] [QRng__init_VERBOSE] YES
    QRng::desc
                             IMPL CHUNKED_CURANDSTATE
                          RNGNAME Philox
                UPLOAD_RNG_STATES NO 
                             seed 0
                           offset 0
                           rngmax 18446744073709551615
                         rngmax/M 18446744073709
                               qr 0x13e03580
        qr.skipahead_event_offset 100000
                             d_qr 0x7f3e24600200

    2024-12-15 19:14:22.802 INFO  [68680] [QSim::simulate@385] sslice {    0,    9,      0,180000000}
    2024-12-15 19:14:22.853 FATAL [68680] [QEvent::setGenstepUpload_NP@230]  gss.desc sslice {    0,    9,      0,180000000}
     gss->ph_offset 0
     gss->ph_count 180000000
     gss->ph_offset + gss->ph_count 180000000(last_rng_state_idx) must be <= max_curand for valid rng_state access
     evt->max_curand -1
     evt->num_curand 0
     evt->max_slot 197000000

    CSGOptiXSMTest: /home/blyth/opticks/qudarap/QEvent.cc:241: int QEvent::setGenstepUpload_NP(const NP*, const sslice*): Assertion `in_range' failed.
    ./cxs_min.sh: line 533: 68680 Aborted                 (core dumped) $bin
    ./cxs_min.sh run error
    P[blyth@localhost opticks]$ 




