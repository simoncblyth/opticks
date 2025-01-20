QSimTest_ALL_initial_shakedown
================================


Lots of FAILs following Philox switch : but seemingly not because of Philox
-------------------------------------------------------------------------------

::

    P[blyth@localhost tests]$ qrng_test
    Philox4_32_10


    P[blyth@localhost tests]$ ~/o/qudarap/tests/QSimTest_ALL.sh

    Fri Dec 13 22:34:51 CST 2024
    Fri Dec 13 22:36:47 CST 2024

     TOTAL : 29 
     PASS  : 12 
     FAIL  : 17 
     === 005 === [ TEST=wavelength_cerenkov /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 005 === ] ***FAIL*** 

        INACTIVE

     === 008 === [ TEST=fill_state_0 /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 008 === ] ***FAIL*** 

     === 009 === [ TEST=fill_state_1 /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 009 === ] ***FAIL*** 

        PYTHON OLD FOLD PATH  


     === 010 === [ TEST=rayleigh_scatter_align /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 010 === ] ***FAIL*** 

         DELIBERATE ASSERT : NOTE TO REVIVE   



     === 012 === [ TEST=hemisphere_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 012 === ] ***FAIL*** 

     === 013 === [ TEST=hemisphere_p_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 013 === ] ***FAIL*** 

     === 014 === [ TEST=hemisphere_x_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 014 === ] ***FAIL*** 

         FIXED PYVISTA ASSUMP


     === 015 === [ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 015 === ] ***FAIL*** 
     === 016 === [ TEST=propagate_at_boundary_p_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 016 === ] ***FAIL*** 
     === 017 === [ TEST=propagate_at_boundary_x_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 017 === ] ***FAIL*** 
     === 018 === [ TEST=propagate_at_boundary /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 018 === ] ***FAIL*** 
     === 019 === [ TEST=propagate_at_boundary_normal_incidence /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 019 === ] ***FAIL*** 

          NOT CLEAR WHY ? ERROR ON PULLBACK  


     === 020 === [ TEST=random_direction_marsaglia /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 020 === ] ***FAIL*** 

     === 021 === [ TEST=lambertian_direction /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 021 === ] ***FAIL*** 

     === 025 === [ TEST=randgaussq_shoot /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 025 === ] ***FAIL*** 

          FIXED PY LEVEL ISSUE



     === 026 === [ TEST=fake_propagate /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 026 === ] ***FAIL*** 
     === 027 === [ TEST=gentorch /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 027 === ] ***FAIL*** 

    P[blyth@localhost tests]$ 




gentorch  : need some genstep setup ? 
------------------------------------------

::

    P[blyth@localhost tests]$ TEST=gentorch /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh
    ...
    === ephoton.sh : TEST gentorch : unset environment : will use C++ defaults in quad4::ephoton for p0
    2024-12-13 23:56:32.455 INFO  [41806] [QSimTest::main@635]  num 100000 type 39 subfold gentorch ni_tranche_size 100000 print_id -1
    2024-12-13 23:56:32.455 INFO  [41806] [QSimTest::generate_photon@391] [ gs_config torch
    2024-12-13 23:56:32.455 INFO  [41806] [QSim::generate_photon@909]  num_photon 0
    2024-12-13 23:56:32.455 FATAL [41806] [QSim::generate_photon@913]  num_photon zero : MUST QEvent::setGenstep before QSim::generate_photon 
    2024-12-13 23:56:32.455 INFO  [41806] [QEvent::gatherPhoton@632] [ evt.num_photon 0 p.sstr (0, 4, 4, ) evt.photon 0
    QU::copy_device_to_host ERROR : device pointer is null 
    2024-12-13 23:56:32.455 FATAL [41806] [QEvent::gatherPhoton@637]  QU::copy_device_to_host photon FAILED  evt->photon N evt->num_photon 0
    === eprd.sh : run error
    P[blyth@localhost tests]$ 



fake_propagate
-------------------

Initial from deliberate SEvt::add_array assert::

    SPrd::fake_prd ni:num_photon 100000 nj:num_bounce 4 num_prd 4
    2024-12-13 23:42:16.280 INFO  [1830] [QSimTest::fake_propagate@493]  num 100000 p (100000, 4, 4, ) bounce_max 4 prd (100000, 4, 2, 4, )
    QSimTest: /home/blyth/opticks/sysrap/SEvt.cc:3631: void SEvt::add_array(const char*, const NP*): Assertion `0' failed.

    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5c37387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5c37387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5c38a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff5c301a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff5c30252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff72f455e in SEvt::add_array (this=0x11842de0, k=0x7ffff76afdd3 "prd0", a=0x11844790) at /home/blyth/opticks/sysrap/SEvt.cc:3631
    #5  0x00007ffff759aa11 in QSim::fake_propagate (this=0x11842a80, prd=0x11844790, type=38) at /home/blyth/opticks/qudarap/QSim.cc:1176
    #6  0x000000000040c7eb in QSimTest::fake_propagate (this=0x7fffffff0560) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:508
    #7  0x000000000040d83b in QSimTest::main (this=0x7fffffff0560) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:706
    #8  0x000000000040ddb3 in main (argc=1, argv=0x7fffffff0d08) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:749
    (gdb) 


Secondary from null fold, when should be topfold (following multi-launch change)::

    Thread 1 "QSimTest" received signal SIGSEGV, Segmentation fault.
    0x00007ffff729defa in NPFold::add_ (this=0x0, k=0x7ffffffef0d0 "prd0.npy", a=0x11844790) at /home/blyth/opticks/sysrap/NPFold.h:1606
    1606        if(verbose_) 
    (gdb) bt
    #0  0x00007ffff729defa in NPFold::add_ (this=0x0, k=0x7ffffffef0d0 "prd0.npy", a=0x11844790) at /home/blyth/opticks/sysrap/NPFold.h:1606
    #1  0x00007ffff729dea4 in NPFold::add (this=0x0, k=0x7ffff76afdd3 "prd0", a=0x11844790) at /home/blyth/opticks/sysrap/NPFold.h:1589
    #2  0x00007ffff72f46cb in SEvt::add_array (this=0x11842de0, k=0x7ffff76afdd3 "prd0", a=0x11844790) at /home/blyth/opticks/sysrap/SEvt.cc:3642
    #3  0x00007ffff759aa11 in QSim::fake_propagate (this=0x11842a80, prd=0x11844790, type=38) at /home/blyth/opticks/qudarap/QSim.cc:1176
    #4  0x000000000040c7eb in QSimTest::fake_propagate (this=0x7fffffff0560) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:508
    #5  0x000000000040d83b in QSimTest::main (this=0x7fffffff0560) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:706
    #6  0x000000000040ddb3 in main (argc=1, argv=0x7fffffff0d08) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:749
    (gdb) 





propagate_at_boundary_s_polarized 
------------------------------------

::

    P[blyth@localhost tests]$ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh
    === ephoton.sh : TEST propagate_at_boundary_s_polarized : unset environment : will use C++ defaults in quad4::ephoton for p0
    2024-12-13 23:14:03.021 INFO  [414046] [main@720] [ TEST propagate_at_boundary_s_polarized
    2024-12-13 23:14:04.731 INFO  [414046] [QRng::initStates@72] initStates<Philox> DO NOTHING 
    2024-12-13 23:14:04.882 INFO  [414046] [QSimTest::EventConfig@600] [ propagate_at_boundary_s_polarized
    2024-12-13 23:14:04.882 INFO  [414046] [QSimTest::EventConfig@615] ] propagate_at_boundary_s_polarized
    2024-12-13 23:14:04.882 INFO  [414046] [SEventConfig::SetDevice@1249] SEventConfig::DescDevice
    name                             : NVIDIA TITAN RTX
    totalGlobalMem_bytes             : 25396576256
    totalGlobalMem_GB                : 23
    HeuristicMaxSlot(VRAM)           : 197276976
    HeuristicMaxSlot(VRAM)/M         : 197
    HeuristicMaxSlot_Rounded(VRAM)   : 197000000
    MaxSlot/M                        : 3

    2024-12-13 23:14:04.882 INFO  [414046] [SEventConfig::SetDevice@1261]  Configured_MaxSlot/M 3 Final_MaxSlot/M 3 HeuristicMaxSlot_Rounded/M 197 changed NO 
     (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2024-12-13 23:14:04.882 INFO  [414046] [QSimTest::main@635]  num 1000000 type 25 subfold propagate_at_boundary_s_polarized ni_tranche_size 100000 print_id -1
    2024-12-13 23:14:04.915 INFO  [414046] [QSimTest::photon_launch_mutate@565]  loaded (1000000, 4, 4, ) from src_subfold hemisphere_s_polarized
    //QSim_photon_launch sim 0x7f9f0a429e00 photon 0x7f9f0de00000 num_photon 1000000 dbg 0x7f9f0a405200 type 25 name propagate_at_boundary_s_polarized 
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ) ) failed with error: 'misaligned address' (/home/blyth/opticks/qudarap/QU.cc:480)

    /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh: line 207: 414046 Aborted                 (core dumped) $bin
    === eprd.sh : run error
    P[blyth@localhost tests]$ 


    (gdb) bt
    #5  0x00007ffff6383669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75d6548 <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f5413 in QU::copy_device_to_host_and_free<sphoton> (h=0x7fff982f6010, d=0x7fffa7e00000, num_items=1000000, label=0x7ffff76b0c9c "QSim::photon_launch_mutate") at /home/blyth/opticks/qudarap/QU.cc:480
    #7  0x00007ffff759b54a in QSim::photon_launch_mutate (this=0x118429e0, photon=0x7fff982f6010, num_photon=1000000, type=25) at /home/blyth/opticks/qudarap/QSim.cc:1105
    #8  0x000000000040cdd7 in QSimTest::photon_launch_mutate (this=0x7fffffff07b0) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:573
    #9  0x000000000040d813 in QSimTest::main (this=0x7fffffff07b0) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:699
    #10 0x000000000040ddb3 in main (argc=1, argv=0x7fffffff0f58) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:749
    (gdb) 





After above fixes
------------------

::

    P[blyth@localhost sysrap]$ ~/o/qudarap/tests/QSimTest_ALL.sh


    Fri Dec 13 23:58:34 CST 2024

    Sat Dec 14 00:00:33 CST 2024

     TOTAL : 29 
     PASS  : 20 
     FAIL  : 9 
     === 005 === [ TEST=wavelength_cerenkov /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 005 === ] ***FAIL*** 
     === 010 === [ TEST=rayleigh_scatter_align /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 010 === ] ***FAIL*** 

               ABOVE TWO ARE DELIBERATE ASSERTS WITH REVIEW AND REVIVE NOTES

     === 015 === [ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 015 === ] ***FAIL*** 
     === 016 === [ TEST=propagate_at_boundary_p_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 016 === ] ***FAIL*** 
     === 017 === [ TEST=propagate_at_boundary_x_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 017 === ] ***FAIL*** 
     === 018 === [ TEST=propagate_at_boundary /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 018 === ] ***FAIL*** 
     === 019 === [ TEST=propagate_at_boundary_normal_incidence /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 019 === ] ***FAIL*** 



     === 026 === [ TEST=fake_propagate /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 026 === ] ***FAIL*** 
     === 027 === [ TEST=gentorch /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 027 === ] ***FAIL*** 

    P[blyth@localhost sysrap]$ 





propagate_at_boundary_s_polarized : 
----------------------------------


::
   
    TEST=propagate_at_boundary_s_polarized ~/o/qudarap/tests/QSimTest.sh 

    2024-12-14 15:53:21.211 INFO  [196766] [SEventConfig::SetDevice@1261]  Configured_MaxSlot/M 3 Final_MaxSlot/M 3 HeuristicMaxSlot_Rounded/M 197 changed NO 
     (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2024-12-14 15:53:21.211 INFO  [196766] [QSimTest::main@635]  num 1000000 type 25 subfold propagate_at_boundary_s_polarized ni_tranche_size 100000 print_id -1
    2024-12-14 15:53:21.265 INFO  [196766] [QSimTest::photon_launch_mutate@565]  loaded (1000000, 4, 4, ) from src_subfold hemisphere_s_polarized
    //QSim_photon_launch sim 0x7fffa4629e00 photon 0x7fffa7e00000 num_photon 1000000 dbg 0x7fffa4605200 type 25 name propagate_at_boundary_s_polarized 
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ) ) failed with error: 'misaligned address' (/home/blyth/opticks/qudarap/QU.cc:480)


    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    0x00007ffff5c37387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff5c37387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff5c38a78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff637689a in __gnu_cxx::__verbose_terminate_handler () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007ffff638236a in __cxxabiv1::__terminate (handler=<optimized out>) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:48
    #4  0x00007ffff63823d5 in std::terminate () at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_terminate.cc:58
    #5  0x00007ffff6382669 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7ffff7a4b338 <typeinfo for QUDA_Exception>, dest=0x7ffff75d5548 <QUDA_Exception::~QUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/download/gcc-11.2.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007ffff75f4413 in QU::copy_device_to_host_and_free<sphoton> (h=0x7fff982f6010, d=0x7fffa7e00000, num_items=1000000, label=0x7ffff76afc9c "QSim::photon_launch_mutate") at /home/blyth/opticks/qudarap/QU.cc:480
    #7  0x00007ffff759a54a in QSim::photon_launch_mutate (this=0x11843180, photon=0x7fff982f6010, num_photon=1000000, type=25) at /home/blyth/opticks/qudarap/QSim.cc:1105
    #8  0x000000000040cdd7 in QSimTest::photon_launch_mutate (this=0x7fffffff3f70) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:573
    #9  0x000000000040d813 in QSimTest::main (this=0x7fffffff3f70) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:699
    #10 0x000000000040ddb3 in main (argc=1, argv=0x7fffffff4718) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:749
    (gdb) 


Old fashioned PIDX=0 debug placing "return 0" at various places in qsim.h reveals that 
things go awry om attempting to use stagr for random collection.


Avoided issue by:

1. removing DEBUG_TAG from qudarap/CMakeLists.txt
2. clean building qudarap (update build was not sufficient to flip the switch)



Issue from DEBUG_TAG being enabled by seeming ctx.tagr not setup
--------------------------------------------------------------------


HUH : removing DEBUG_TAG from qudarap/CMakeLists.txt not working, perhaps need clean build ? 


* :doc:`DEBUG_TAG_not_easy_to_turn_off_within_debug_build`

::

    1095 #if !defined(PRODUCTION) && defined(DEBUG_TAG)
    1096     stagr& tagr = ctx.tagr ;
    1097     tagr.add( stag_at_burn_sf_sd, u_boundary_burn);
    1098     tagr.add( stag_at_ref,  u_reflect);
    1099 #endif




After DEBUG_TAG switch off for qudarap : down to 6/29 FAILs
-----------------------------------------------------------------

::

    Sat Dec 14 18:46:03 CST 2024
    Sat Dec 14 18:48:06 CST 2024

     TOTAL : 29 
     PASS  : 23 
     FAIL  : 6 
     === 005 === [ TEST=wavelength_cerenkov /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 005 === ] ***FAIL*** 
     === 010 === [ TEST=rayleigh_scatter_align /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 010 === ] ***FAIL*** 
     === 018 === [ TEST=propagate_at_boundary /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 018 === ] ***FAIL*** 
     === 019 === [ TEST=propagate_at_boundary_normal_incidence /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 019 === ] ***FAIL*** 


            ABOVE FOUR ARE DELIBERATE ASSERTS WITH REVIEW AND REVIVE NOTES



     === 026 === [ TEST=fake_propagate /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 026 === ] ***FAIL*** 

            EMPTY FOLD ? 


     === 027 === [ TEST=gentorch /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 027 === ] ***FAIL*** 






gentorch : fixed by updated genstep handling
-------------------------------------------------

::

    P[blyth@localhost qudarap]$ TEST=gentorch /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh
    === ephoton.sh : TEST gentorch : unset environment : will use C++ defaults in quad4::ephoton for p0
    2024-12-14 18:54:07.069 INFO  [113267] [main@734] [ TEST gentorch
    2024-12-14 18:54:08.780 INFO  [113267] [QRng::initStates@72] initStates<Philox> DO NOTHING 
    2024-12-14 18:54:08.923 INFO  [113267] [QSimTest::EventConfig@614] [ gentorch
    2024-12-14 18:54:08.923 INFO  [113267] [QSimTest::EventConfig@629] ] gentorch
    2024-12-14 18:54:08.923 INFO  [113267] [QSimTest::main@649]  num 100000 type 39 subfold gentorch ni_tranche_size 100000 print_id -1
    2024-12-14 18:54:08.923 INFO  [113267] [QSimTest::generate_photon@392] [ gs_config torch
    2024-12-14 18:54:08.924 INFO  [113267] [QSim::generate_photon@909]  num_photon 0
    2024-12-14 18:54:08.924 FATAL [113267] [QSim::generate_photon@913]  num_photon zero : MUST QEvent::setGenstep before QSim::generate_photon 
    2024-12-14 18:54:08.924 INFO  [113267] [QEvent::gatherPhoton@632] [ evt.num_photon 0 p.sstr (0, 4, 4, ) evt.photon 0
    QU::copy_device_to_host ERROR : device pointer is null 
    2024-12-14 18:54:08.924 FATAL [113267] [QEvent::gatherPhoton@637]  QU::copy_device_to_host photon FAILED  evt->photon N evt->num_photon 0
    === eprd.sh : run error
    P[blyth@localhost qudarap]$ 
    P[blyth@localhost qudarap]$ 



::

    (gdb) bt
    #0  0x00007ffff6ff04fb in raise () from /lib64/libpthread.so.0
    #1  0x00007ffff75e7956 in QEvent::gatherPhoton (this=0x118095b0, p=0x1180b020) at /home/blyth/opticks/qudarap/QEvent.cc:643
    #2  0x00007ffff75e7b13 in QEvent::gatherPhoton (this=0x118095b0) at /home/blyth/opticks/qudarap/QEvent.cc:652
    #3  0x000000000040bc7e in QSimTest::generate_photon (this=0x7fffffff0aa0) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:402
    #4  0x000000000040d822 in QSimTest::main (this=0x7fffffff0aa0) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:679
    #5  0x000000000040df45 in main (argc=1, argv=0x7fffffff1248) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:763
    (gdb) 



::

    388 void QSimTest::generate_photon()
    389 {
    390     const char* gs_config = ssys::getenvvar("GS_CONFIG", "torch" );
    391 
    392     LOG(info) << "[ gs_config " << gs_config ;
    393     const NP* gs = SEvent::MakeDemoGenstep(gs_config);
    394 
    395     SEvt* evt = SEvt::Create(SEvt::EGPU) ;
    396     assert(evt);
    397 
    398     evt->addGenstep(gs);
    399 
    400     qs->generate_photon();
    401 
    402     NP* p = qs->event->gatherPhoton();
    403     p->save("$FOLD/p.npy");
    404 
    405     LOG(info) << "]" ;
    406 }







Down to 4 fails : which are the deliberate asserts : indicating need to be reimplemented
----------------------------------------------------------------------------------------

::

    Sat Dec 14 21:55:39 CST 2024
    Sat Dec 14 21:57:43 CST 2024

     TOTAL : 29 
     PASS  : 25 
     FAIL  : 4 
     === 005 === [ TEST=wavelength_cerenkov /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 005 === ] ***FAIL*** 
     === 010 === [ TEST=rayleigh_scatter_align /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 010 === ] ***FAIL*** 
     === 018 === [ TEST=propagate_at_boundary /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 018 === ] ***FAIL*** 
     === 019 === [ TEST=propagate_at_boundary_normal_incidence /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 019 === ] ***FAIL*** 




wavelength_cerenkov
---------------------

::

    P[blyth@localhost opticks]$ opticks-f wavelength_cerenkov
    ./qudarap/QSimLaunch.hh:    static constexpr const char* WAVELENGTH_CERENKOV_ = "wavelength_cerenkov" ; 
    ./qudarap/tests/QCtxTest.py:        name = "wavelength_cerenkov"
    ./qudarap/tests/QSimTest.sh:#test=wavelength_cerenkov         ### non-active moved to QSim_dbg.cu 
    ./qudarap/tests/QSimTest_ALL.sh:wavelength_cerenkov
    P[blyth@localhost opticks]$ 




Rerun 2025/01/20 : 3/25 FAILs
--------------------------------

::

    Mon Jan 20 14:51:33 CST 2025

     TOTAL : 25 
     PASS  : 22 
     FAIL  : 3 
     === 013 === [ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 013 === ] ***FAIL*** 
     === 014 === [ TEST=propagate_at_boundary_p_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 014 === ] ***FAIL*** 
     === 015 === [ TEST=propagate_at_boundary_x_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 
     === 015 === ] ***FAIL*** 



::

    P[blyth@localhost tests]$ TEST=propagate_at_boundary_s_polarized /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh 

    //QSim_photon_launch sim 0x7f622e629e00 photon 0x7f6231e00000 num_photon 1000000 dbg 0x7f622e605200 type 25 name propagate_at_boundary_s_polarized 
    //QSim_photon_launch post launch cudaDeviceSynchronize 
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ) ) failed with error: 'misaligned address' (/home/blyth/opticks/qudarap/QU.cc:514)

    /data/blyth/junotop/opticks/qudarap/tests/QSimTest.sh: line 210: 395000 Aborted                 (core dumped) $bin




