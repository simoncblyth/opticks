Fabstep Targetting
====================

4 of 6 remaining test fails are from the same target_acquired assert::


    97% tests passed, 6 tests failed out of 234

    Total Test time (real) = 116.68 sec

    The following tests FAILED:
        208 - OptiXRapTest.OEventTest (OTHER_FAULT)
        209 - OptiXRapTest.OInterpolationTest (Failed)
        213 - OKOPTest.OpSeederTest (OTHER_FAULT)
        220 - OKTest.VizTest (OTHER_FAULT)

        222 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        223 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output
    simon:opticks blyth$ 


::


    (lldb) f 4
    frame #4: 0x0000000101a20c99 libNPY.dylib`GenstepNPY::addStep(this=0x0000000108a4d430, verbose=false) + 473 at GenstepNPY.cpp:77
       74                       << brief()
       75                       ;
       76   
    -> 77       assert(target_acquired);
       78   
       79       assert(m_npy && m_npy->hasData());
       80   


    (lldb) f 5
    frame #5: 0x0000000101a2072f libNPY.dylib`FabStepNPY::init(this=0x0000000108a4d430) + 111 at FabStepNPY.cpp:20
       17       {   
       18           setMaterialLine(i*10);   
       19           setNumPhotons(m_num_photons_per_step); 
    -> 20           addStep();
       21       } 
       22   }

    (lldb) f 8
    frame #8: 0x0000000101849119 libOpticksGeometry.dylib`OpticksGen::makeFabstep(this=0x0000000108a4b320) + 73 at OpticksGen.cc:173
       170  
       171  FabStepNPY* OpticksGen::makeFabstep()
       172  {
    -> 173      FabStepNPY* fabstep = new FabStepNPY(FABRICATED, 10, 10 );
       174  
       175      const char* material = m_ok->getDefaultMaterial();
       176      fabstep->setMaterial(material);
    (lldb) bt
    * thread #1: tid = 0x730353, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8d3cf9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101a20c99 libNPY.dylib`GenstepNPY::addStep(this=0x0000000108a4d430, verbose=false) + 473 at GenstepNPY.cpp:77
        frame #5: 0x0000000101a2072f libNPY.dylib`FabStepNPY::init(this=0x0000000108a4d430) + 111 at FabStepNPY.cpp:20
        frame #6: 0x0000000101a20696 libNPY.dylib`FabStepNPY::FabStepNPY(this=0x0000000108a4d430, genstep_type=32768, num_step=10, num_photons_per_step=10) + 70 at FabStepNPY.cpp:10
        frame #7: 0x0000000101a20777 libNPY.dylib`FabStepNPY::FabStepNPY(this=0x0000000108a4d430, genstep_type=32768, num_step=10, num_photons_per_step=10) + 39 at FabStepNPY.cpp:11
      * frame #8: 0x0000000101849119 libOpticksGeometry.dylib`OpticksGen::makeFabstep(this=0x0000000108a4b320) + 73 at OpticksGen.cc:173
        frame #9: 0x0000000101848d72 libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x0000000108a4b320) + 690 at OpticksGen.cc:74
        frame #10: 0x0000000101848a85 libOpticksGeometry.dylib`OpticksGen::init(this=0x0000000108a4b320) + 21 at OpticksGen.cc:37
        frame #11: 0x0000000101848a63 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108a4b320, hub=0x00007fff5fbfec30) + 131 at OpticksGen.cc:32
        frame #12: 0x0000000101848aad libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108a4b320, hub=0x00007fff5fbfec30) + 29 at OpticksGen.cc:33
        frame #13: 0x0000000101846026 libOpticksGeometry.dylib`OpticksHub::init(this=0x00007fff5fbfec30) + 118 at OpticksHub.cc:96
        frame #14: 0x0000000101845f00 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfec30, ok=0x00007fff5fbfeca0) + 416 at OpticksHub.cc:81
        frame #15: 0x00000001018460dd libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfec30, ok=0x00007fff5fbfeca0) + 29 at OpticksHub.cc:83
        frame #16: 0x0000000100006374 OEventTest`main(argc=1, argv=0x00007fff5fbfee58) + 612 at OEventTest.cc:37
        frame #17: 0x00007fff8a48b5fd libdyld.dylib`start + 1
        frame #18: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) 


