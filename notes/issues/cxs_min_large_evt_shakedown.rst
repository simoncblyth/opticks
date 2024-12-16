cxs_min_large_evt_shakedown
============================


::

    TEST=large_evt ~/o/cxs_min.sh 


WIP : check large_evt on Titan+Ada
--------------------------------------

1. DONE : update /cvmfs with GEOM J_2024nov27
2. DONE: update opticks 
3. DONE : Skip SCurandStateMonolithicTest for RNG_PHILOX
4. DONE: get opticks-t to pass
5. TODO: get ~/o/qudarap/tests/QSimTest_ALL.sh to pass 



Issue 2 : FIXED : opticks-t fails from max_photon 1 billion with OOM 
-------------------------------------------------------------------------

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



::

    0978 void QEvent::device_alloc_photon()
     979 {
     980     LOG_IF(info, LIFECYCLE) ;
     981     SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
     982 
     983     LOG(LEVEL)
     984         << " evt.max_slot   " << evt->max_slot
     985         << " evt.max_photon " << evt->max_photon
     986         << " evt.num_photon " << evt->num_photon
     987 #ifndef PRODUCTION
     988         << " evt.num_record " << evt->num_record
     989         << " evt.num_rec    " << evt->num_rec
     990         << " evt.num_seq    " << evt->num_seq
     991         << " evt.num_prd    " << evt->num_prd
     992         << " evt.num_tag    " << evt->num_tag
     993         << " evt.num_flat   " << evt->num_flat
     994 #endif
     995         ;
     996 
     997     evt->photon  = evt->max_slot > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot, "QEvent::device_alloc_photon/max_slot*sizeof(sphoton)" ) : nullptr ;
     998 
     999 #ifndef PRODUCTION
    1000     evt->record  = evt->max_record > 0 ? QU::device_alloc_zero<sphoton>( evt->max_slot * evt->max_record, "max_slot*max_record*sizeof(sphoton)" ) : nullptr ;
    1001     evt->rec     = evt->max_rec    > 0 ? QU::device_alloc_zero<srec>(    evt->max_slot * evt->max_rec   , "max_slot*max_rec*sizeof(srec)"    ) : nullptr ;
    1002     evt->prd     = evt->max_prd    > 0 ? QU::device_alloc_zero<quad2>(   evt->max_slot * evt->max_prd   , "max_slot*max_prd*sizeof(quad2)"    ) : nullptr ;
    1003     evt->seq     = evt->max_seq   == 1 ? QU::device_alloc_zero<sseq>(    evt->max_slot                  , "max_slot*sizeof(sseq)"    ) : nullptr ;
    1004     evt->tag     = evt->max_tag   == 1 ? QU::device_alloc_zero<stag>(    evt->max_slot                  , "max_slot*sizeof(stag)"    ) : nullptr ;
    1005     evt->flat    = evt->max_flat  == 1 ? QU::device_alloc_zero<sflat>(   evt->max_slot                  , "max_slot*sizeof(sflat)"   ) : nullptr ;
    1006 #endif
    1007 
    1008     LOG(LEVEL) << desc() ;
    1009     LOG(LEVEL) << desc_alloc() ;
    1010 }



    P[blyth@localhost tests]$ opticks-f QU::alloc
    ./CSGOptiX/CSGOptiX.cc:    QU::alloc = SEventConfig::ALLOC ; 
    ./qudarap/QEvent.cc:    salloc* alloc = QU::alloc ; 
    ./qudarap/QEvent.cc:    SetAllocMeta( QU::alloc, evt );   // do this first as memory errors likely to happen in following lines
    ./qudarap/QU.cc:salloc* QU::alloc = nullptr ;   // used to monitor allocations, instanciated in CSGOptiX::Create
    P[blyth@localhost opticks]$ 



DONE : Ordinarily allocating for max_slot is appropriate but not for fake_propagate which does step point record recording
--------------------------------------------------------------------------------------------------------------------------------

Fixed QSimTest:fake_propagate OOM by MaxSlot reduction for debug array runinng from QSimTest::EventConfig, improved OOM reporting. 

::

    SPrd::fake_prd ni:num_photon 100000 nj:num_bounce 4 num_prd 4
    2024-12-16 10:32:24.503 INFO  [310604] [QSimTest::fake_propagate@517]  num 100000 p (100000, 4, 4, ) bounce_max 4 prd (100000, 4, 2, 4, )
    2024-12-16 10:32:24.542 ERROR [310604] [QU::_cudaMalloc@260] save salloc record to /data/blyth/opticks/GEOM/J_2024nov27/QSimTest
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (max_slot*max_record*sizeof(sphoton) ) failed with error: 'out of memory' (/home/blyth/opticks/qudarap/QU.cc:253)
    [salloc::desc alloc.size 7 label.size 7
    [salloc.meta
    evt.max_curand:1000000000
    evt.max_slot:197000000
    evt.max_photon:1000000
    evt.num_photon:100000
    evt.max_curand/M:1000
    evt.max_slot/M:197
    evt.max_photon/M:1
    evt.num_photon/M:0
    evt.max_record:5
    evt.max_rec:0
    evt.max_seq:1
    evt.max_prd:0
    evt.max_tag:0
    evt.max_flat:0
    evt.num_record:500000
    evt.num_rec:0
    evt.num_seq:100000
    evt.num_prd:0
    evt.num_tag:0
    evt.num_flat:0
    ]salloc.meta

         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9            

         [            256           1         256           0]       0.00       0.00 QEvent::QEvent/sevent
         [             64           1          64           0]       0.00       0.00 QSim::init.sim
         [       12800000      400000          32           0]       0.01       0.02 QSim::UploadFakePRD/d_prd
         [             96           1          96           0]       0.00       0.00 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [        4000000     1000000           4           0]       0.00       0.01 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [    12608000000   197000000          64           0]      12.61      16.66 QEvent::device_alloc_photon/max_slot*sizeof(sphoton)
         [    63040000000   985000000          64           0]      63.04      83.31 max_slot*max_record*sizeof(sphoton)

     tot      75664800416                                           75.66
    ]salloc::desc


    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    (gdb) 
    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5c18387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5c19a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff635789a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff636336a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff63633d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6363669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75ef9dc <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f3103 in QU::_cudaMalloc (p2p=0x7ffffffefb40, size=63040000000, label=0x7ffff76cfcf8 "max_slot*max_record*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:267
    #7  0x00007ffff7606539 in QU::device_alloc_zero<sphoton> (num_items=985000000, label=0x7ffff76cfcf8 "max_slot*max_record*sizeof(sphoton)") at /home/blyth/opticks/qudarap/QU.cc:343
    #8  0x00007ffff75e9daf in QEvent::device_alloc_photon (this=0x11809e50) at /home/blyth/opticks/qudarap/QEvent.cc:1000
    #9  0x00007ffff75e9926 in QEvent::setNumPhoton (this=0x11809e50, num_photon=100000) at /home/blyth/opticks/qudarap/QEvent.cc:952
    #10 0x00007ffff75e5a30 in QEvent::setInputPhotonAndUpload (this=0x11809e50) at /home/blyth/opticks/qudarap/QEvent.cc:461
    #11 0x00007ffff75e5375 in QEvent::setGenstepUpload (this=0x11809e50, qq0=0x13cac3a0, gs_start=0, gs_stop=1) at /home/blyth/opticks/qudarap/QEvent.cc:366
    #12 0x00007ffff75e450b in QEvent::setGenstepUpload_NP (this=0x11809e50, gs_=0x13cac260, gss_=0x0) at /home/blyth/opticks/qudarap/QEvent.cc:214
    #13 0x00007ffff75e3f41 in QEvent::setGenstepUpload_NP (this=0x11809e50, gs_=0x13cac260) at /home/blyth/opticks/qudarap/QEvent.cc:180
    #14 0x00007ffff75b4c24 in QSim::fake_propagate (this=0x11809da0, prd=0x1180ba10, type=38) at /home/blyth/opticks/qudarap/QSim.cc:1189
    #15 0x000000000040caa3 in QSimTest::fake_propagate (this=0x7fffffff3e60) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:532
    #16 0x000000000040dc85 in QSimTest::main (this=0x7fffffff3e60) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:743
    #17 0x000000000040e1fd in main (argc=1, argv=0x7fffffff4608) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:786
    (gdb) 


Adjusted to not use the production relevant heuristic max_slot with debug array running in QSimTest::EventConfig::

    521 void QSimTest::EventConfig(unsigned type, const SPrd* prd )  // static
    522 {
    523     SEvt* sev = SEvt::Get_EGPU();
    524     LOG_IF(fatal, sev != nullptr ) << "QSimTest::EventConfig must be done prior to instanciating SEvt, eg for fake_propagate bounce consistency " ;
    525     assert(sev == nullptr);
    526 
    527     LOG(LEVEL) << "[ " <<  QSimLaunch::Name(type) ;
    528     if( type == FAKE_PROPAGATE )
    529     {
    530         LOG(LEVEL) << prd->desc() ;
    531         int maxbounce = prd->getNumBounce();
    532 
    533         SEventConfig::SetMaxBounce(maxbounce);
    534         SEventConfig::SetEventMode("DebugLite");
    535         SEventConfig::Initialize();
    536 
    537         SEventConfig::SetMaxGenstep(1);    // FAKE_PROPAGATE starts from input photons but uses a single placeholder genstep 
    538 
    539         unsigned mx = 1000000 ;
    540         SEventConfig::SetMaxPhoton(mx);   // used for QEvent buffer sizing 
    541         SEventConfig::SetMaxSlot(mx);
    542         // greatly reduced MaxSlot as debug arrays in use
    543 
    544         LOG(LEVEL) << " SEventConfig::Desc " << SEventConfig::Desc() ;
    545     }
    546     LOG(LEVEL) << "] " <<  QSimLaunch::Name(type) ;
    547 }







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




