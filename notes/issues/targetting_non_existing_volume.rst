Targetting Non Existing Volume
=================================


With partial geometry need way to change default target volume 3153::

    2017-05-20 12:03:21.642 WARN  [2424808] [*GMesh::getTransform@855] GMesh::getTransform out of bounds  m_num_solids 11 index 3153
    2017-05-20 12:03:21.642 INFO  [2424808] [OpticksGen::targetGenstep@125] OpticksGen::targetGenstep setting frame 3153 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000




::

    (lldb) bt
    * thread #1: tid = 0x24e6c0, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8d3cf9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101e65502 libGGeo.dylib`GMesh::getTransform(this=0x0000000108b5f760, index=3153) + 66 at GMesh.cc:854
        frame #5: 0x0000000101eacb30 libGGeo.dylib`GGeo::getTransform(this=0x0000000105a138a0, index=3153) + 128 at GGeo.cc:1553
        frame #6: 0x0000000101fdc85c libOpticksGeometry.dylib`OpticksGen::targetGenstep(this=0x0000000108b6aae0, gs=0x0000000108b6ab20) + 444 at OpticksGen.cc:124
        frame #7: 0x0000000101fdc214 libOpticksGeometry.dylib`OpticksGen::makeTorchstep(this=0x0000000108b6aae0) + 52 at OpticksGen.cc:182
        frame #8: 0x0000000101fdbdce libOpticksGeometry.dylib`OpticksGen::initInputGensteps(this=0x0000000108b6aae0) + 606 at OpticksGen.cc:74
      * frame #9: 0x0000000101fdbb35 libOpticksGeometry.dylib`OpticksGen::init(this=0x0000000108b6aae0) + 21 at OpticksGen.cc:37
        frame #10: 0x0000000101fdbb13 libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108b6aae0, hub=0x0000000105a0ceb0) + 131 at OpticksGen.cc:32
        frame #11: 0x0000000101fdbb5d libOpticksGeometry.dylib`OpticksGen::OpticksGen(this=0x0000000108b6aae0, hub=0x0000000105a0ceb0) + 29 at OpticksGen.cc:33
        frame #12: 0x0000000101fd90d6 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105a0ceb0) + 118 at OpticksHub.cc:96
        frame #13: 0x0000000101fd8fb0 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105a0ceb0, ok=0x0000000105921a30) + 416 at OpticksHub.cc:81
        frame #14: 0x0000000101fd918d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105a0ceb0, ok=0x0000000105921a30) + 29 at OpticksHub.cc:83
        frame #15: 0x00000001039481e6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe918, argc=21, argv=0x00007fff5fbfe9f8, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #16: 0x000000010394864b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe918, argc=21, argv=0x00007fff5fbfe9f8, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #17: 0x000000010000a93d OKTest`main(argc=21, argv=0x00007fff5fbfe9f8) + 1373 at OKTest.cc:60
        frame #18: 0x00007fff8a48b5fd libdyld.dylib`start + 1
        frame #19: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) 


