QSimTest_rng_sequence_with_skipahead_FAIL_from_no_gs
========================================================


rng_sequence_with_skipahead : REMOVED
-------------------------------------------

* bizarrely was calling qsim->simulate with no genstep setup : REMOVED IT

::

    P[blyth@localhost ~]$ TEST=rng_sequence_with_skipahead ~/o/qudarap/tests/QSimTest.sh
    === ephoton.sh : TEST rng_sequence_with_skipahead : unset environment : will use C++ defaults in quad4::ephoton for p0
    2024-12-13 17:23:39.928 INFO  [281873] [main@737] [ TEST rng_sequence_with_skipahead
    2024-12-13 17:23:41.623 INFO  [281873] [QRng::initStates@80] initStates<XORWOW> LoadAndUpload and set_uploaded_states 
    QRng::LoadAndUpload complete YES rngmax/M 3 rngmax 3000000 digest c5a80f522e9393efe0302b916affda06

    2024-12-13 17:23:42.786 INFO  [281873] [SEventConfig::SetDevice@1261]  Configured_MaxSlot/M 3 Final_MaxSlot/M 3 HeuristicMaxSlot_Rounded/M 197 changed NO 
     (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    2024-12-13 17:23:42.786 INFO  [281873] [QSimTest::main@651]  num 1000000 type 2 subfold rng_sequence_with_skipahead ni_tranche_size 100000 print_id -1
    2024-12-13 17:23:42.786 INFO  [281873] [QSimTest::rng_sequence_with_skipahead@168]  eventID_key QSimTest__rng_sequence_with_skipahead__eventID eventID 0
    2024-12-13 17:23:42.786 FATAL [281873] [QSim::simulate@373]  gs_null YES
    QSimTest: /home/blyth/opticks/qudarap/QSim.cc:374: double QSim::simulate(int, bool): Assertion `!gs_null' failed.
    /home/blyth/o/qudarap/tests/QSimTest.sh: line 207: 281873 Aborted                 (core dumped) $bin
    === eprd.sh : run error
    P[blyth@localhost ~]$ 


rng_sequence : FIXED 
-----------------------

::

    P[blyth@localhost sysrap]$ TEST=rng_sequence ~/o/qudarap/tests/QSimTest.sh dbg
    === ephoton.sh : TEST rng_sequence : unset environment : will use C++ defaults in quad4::ephoton for p0
    ...
    2024-12-13 17:28:08.038 INFO  [288951] [QSimTest::main@651]  num 1000000 type 1 subfold rng_sequence ni_tranche_size 100000 print_id -1
    QSim::rng_sequence ni 1000000 ni_tranche_size 100000 skipahead NO  num_tranche 10 reldir rng_sequence_f_ni1000000_nj16_nk16_tranche100000 nj 16 nk 16 nv(nj*nk) 256 size(ni_tranche_size*nv) 25600000 typecode f
      0         0                                                 rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy
    //QSim_rng_sequence ni 100000 nv 256 id_offset 0 skipahead 0  
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost ) ) failed with error: 'an illegal memory access was encountered' (/home/blyth/opticks/qudarap/QU.cc:480)

    Thread 1 "QSimTest" received signal SIGABRT, Aborted.
    (gdb) bt
    ...
    #6  0x00007ffff76106a5 in QU::copy_device_to_host_and_free<float> (h=0x7fff8be57010, d=0x7fff82000000, num_items=25600000, label=0x7ffff76ceb37 "QSim::rng_sequence:num_rng") at /home/blyth/opticks/qudarap/QU.cc:480
                                                                       ^^^^^^^^^^^^^^^^ 
                                                                       looks like device pointer ? 

    #7  0x00007ffff75d7d9f in QSim::rng_sequence<float> (this=0x1180c630, seq=0x7fff8be57010, ni_tranche=100000, nv=256, id_offset=0, skipahead=false) at /home/blyth/opticks/qudarap/QSim.cc:700
    #8  0x00007ffff75cd987 in QSim::rng_sequence<float> (this=0x1180c630, dir=0x4b508f "$FOLD", ni=1000000, nj=16, nk=16, ni_tranche_size=100000, skipahead=false) at /home/blyth/opticks/qudarap/QSim.cc:767
    #9  0x000000000040ad14 in QSimTest::rng_sequence (this=0x7fffffff3fa0, ni=1000000, ni_tranche_size_=100000) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:154
    #10 0x000000000040d73d in QSimTest::main (this=0x7fffffff3fa0) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:661
    #11 0x000000000040deef in main (argc=1, argv=0x7fffffff4748) at /home/blyth/opticks/qudarap/tests/QSimTest.cc:766
    (gdb) 



wavelength_scintillation : FIXED
-----------------------------------





