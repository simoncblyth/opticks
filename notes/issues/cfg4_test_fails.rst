Test Fails
=============

::



    97% tests passed, 7 tests failed out of 235

    Total Test time (real) = 118.38 sec

    The following tests FAILED:
         39 - NPYTest.NOpenMeshCombineTest (OTHER_FAULT)
         40 - NPYTest.NPolygonizerTest (OTHER_FAULT)
         41 - NPYTest.NCSGBSPTest (OTHER_FAULT)
         75 - NPYTest.NConvexPolyhedronTest (OTHER_FAULT)
        175 - GGeoTest.GSceneTest (SEGFAULT)                 ## MISSING PATH ERROR, FIXED
        210 - OptiXRapTest.OInterpolationTest (Failed)
        224 - cfg4Test.CTestDetectorTest (OTHER_FAULT)       ## SEE BELOW
    Errors while running CTest

    simon:~ blyth$ date
    Thu Jun 22 18:52:52 CST 2017


::

    simon:issues blyth$ CTestDetectorTest
    2017-06-22 18:53:31.070 INFO  [348725] [main@42] CTestDetectorTest
    2017-06-22 18:53:31.246 INFO  [348725] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-22 18:53:31.359 INFO  [348725] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-22 18:53:31.440 INFO  [348725] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-22 18:53:31.440 INFO  [348725] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-22 18:53:31.580 INFO  [348725] [*GMergedMesh::combine@122] GMergedMesh::combine making new mesh  index 0 solids 1 verbosity 1
    ...
    2017-06-22 18:53:31.580 INFO  [348725] [GSolid::Dump@219] GMergedMesh::combine (source solids) numSolid 1
    2017-06-22 18:53:31.580 INFO  [348725] [GNode::dump@205] mesh.numSolids 0 mesh.ce.0 gfloat4      0.000      0.000      0.000    300.000 
    2017-06-22 18:53:31.581 FATAL [348725] [GMergedMesh::mergeSolidIdentity@482] GMergedMesh::mergeSolidIdentity mismatch  nodeIndex 0 m_cur_solid 6
    2017-06-22 18:53:31.581 INFO  [348725] [GMergedMesh::dumpSolids@659] GMergedMesh::combine (combined result)  ce0 gfloat4      0.000      0.000      0.000    300.000 
        0 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
        1 ce             gfloat4      0.000      0.000    -18.997    149.997  bb bb min   -100.288   -100.288   -168.995  max    100.288    100.288    131.000 
        2 ce             gfloat4      0.000      0.000    -18.247    146.247  bb bb min    -97.288    -97.288   -164.495  max     97.288     97.288    128.000 
        3 ce             gfloat4      0.005      0.004     91.998     98.143  bb bb min    -98.138    -98.139     55.996  max     98.148     98.147    128.000 
        4 ce             gfloat4      0.000      0.000     13.066     98.143  bb bb min    -98.143    -98.143    -30.000  max     98.143     98.143     56.131 
        5 ce             gfloat4      0.000      0.000    -81.500     83.000  bb bb min    -27.500    -27.500   -164.500  max     27.500     27.500      1.500 
        6 ce             gfloat4      0.000      0.000      0.000    300.000  bb bb min   -300.000   -300.000   -300.000  max    300.000    300.000    300.000 
        0 ni[nf/nv/nidx/pidx] (  0,  0,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,  5,  0,  0) 
        1 ni[nf/nv/nidx/pidx] (720,362,  1,  0)  id[nidx,midx,bidx,sidx]  (  1,  4,  1,  0) 
        2 ni[nf/nv/nidx/pidx] (720,362,  2,  1)  id[nidx,midx,bidx,sidx]  (  2,  3,  2,  0) 
        3 ni[nf/nv/nidx/pidx] (960,482,  3,  2)  id[nidx,midx,bidx,sidx]  (  3,  0,  3,  0) 
        4 ni[nf/nv/nidx/pidx] (576,288,  4,  2)  id[nidx,midx,bidx,sidx]  (  4,  1,  4,  0) 
        5 ni[nf/nv/nidx/pidx] ( 96, 50,  5,  2)  id[nidx,midx,bidx,sidx]  (  5,  2,  4,  0) 
        6 ni[nf/nv/nidx/pidx] ( 12, 24,  0,4294967295)  id[nidx,midx,bidx,sidx]  (  0,1000,  0,  0) 
    Assertion failed: (m_bndlib), function registerBoundaries, file /Users/blyth/opticks/ggeo/GParts.cc, line 621.
    Abort trap: 6
    simon:issues blyth$ 


    (lldb) bt
    * thread #1: tid = 0x5553a, 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8ddca35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff94b1ab1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff94ae49bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100d49da0 libGGeo.dylib`GParts::registerBoundaries(this=0x000000010cc13420) + 96 at GParts.cc:621
        frame #5: 0x0000000100d49c18 libGGeo.dylib`GParts::close(this=0x000000010cc13420) + 264 at GParts.cc:611
        frame #6: 0x0000000100d75d18 libGGeo.dylib`GGeoTest::createPmtInBox(this=0x000000010b567970) + 1368 at GGeoTest.cc:187
        frame #7: 0x0000000100d753be libGGeo.dylib`GGeoTest::create(this=0x000000010b567970) + 126 at GGeoTest.cc:109
        frame #8: 0x0000000100d7529d libGGeo.dylib`GGeoTest::modifyGeometry(this=0x000000010b567970) + 157 at GGeoTest.cc:81
        frame #9: 0x0000000100d99f7c libGGeo.dylib`GGeo::modifyGeometry(this=0x0000000107a2ae00, config=0x0000000000000000) + 668 at GGeo.cc:780
        frame #10: 0x0000000101112824 libOpticksGeometry.dylib`OpticksGeometry::modifyGeometry(this=0x0000000107a2bfd0) + 868 at OpticksGeometry.cc:264
        frame #11: 0x0000000101111d6c libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000107a2bfd0) + 572 at OpticksGeometry.cc:201
        frame #12: 0x0000000101115e59 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x00007fff5fbfea88) + 409 at OpticksHub.cc:243
        frame #13: 0x0000000101114fed libOpticksGeometry.dylib`OpticksHub::init(this=0x00007fff5fbfea88) + 77 at OpticksHub.cc:94
        frame #14: 0x0000000101114ef0 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfea88, ok=0x00007fff5fbfeaf8) + 416 at OpticksHub.cc:81
        frame #15: 0x00000001011150cd libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x00007fff5fbfea88, ok=0x00007fff5fbfeaf8) + 29 at OpticksHub.cc:83
        frame #16: 0x000000010000d026 CTestDetectorTest`main(argc=1, argv=0x00007fff5fbfee08) + 950 at CTestDetectorTest.cc:48
        frame #17: 0x00007fff91ba05fd libdyld.dylib`start + 1
        frame #18: 0x00007fff91ba05fd libdyld.dylib`start + 1


::

    (lldb) bt
    * thread #1: tid = 0x55be8, 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff9672d866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8ddca35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff94b1ab1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff94ae49bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100d36a6d libGGeo.dylib`GSurLib::examineSolidBndSurfaces(this=0x000000010c436de0) + 2109 at GSurLib.cc:147
        frame #5: 0x0000000100d3621d libGGeo.dylib`GSurLib::close(this=0x000000010c436de0) + 29 at GSurLib.cc:93
        frame #6: 0x00000001015e0197 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010c436e40) + 247 at CDetector.cc:244
        frame #7: 0x000000010155a714 libcfg4.dylib`CGeometry::init(this=0x000000010c436b70) + 1476 at CGeometry.cc:73
        frame #8: 0x000000010155a140 libcfg4.dylib`CGeometry::CGeometry(this=0x000000010c436b70, hub=0x00007fff5fbfea88) + 112 at CGeometry.cc:39
        frame #9: 0x000000010155a77d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010c436b70, hub=0x00007fff5fbfea88) + 29 at CGeometry.cc:40
        frame #10: 0x0000000101601146 libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe9d0, hub=0x00007fff5fbfea88) + 214 at CG4.cc:122
        frame #11: 0x00000001016016dd libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe9d0, hub=0x00007fff5fbfea88) + 29 at CG4.cc:144
        frame #12: 0x000000010000d03e CTestDetectorTest`main(argc=1, argv=0x00007fff5fbfee08) + 974 at CTestDetectorTest.cc:50
        frame #13: 0x00007fff91ba05fd libdyld.dylib`start + 1
        frame #14: 0x00007fff91ba05fd libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x0000000100d36a6d libGGeo.dylib`GSurLib::examineSolidBndSurfaces(this=0x000000010c436de0) + 2109 at GSurLib.cc:147
       144                        << " lv " << ( lv ? lv : "NULL" )
       145                        ;
       146  
    -> 147          assert( node == i );
       148  
       149  
       150          //unsigned mesh = id.y ;
    (lldb) p node
    (unsigned int) $0 = 0
    (lldb) p i
    (unsigned int) $1 = 6
    (lldb) 

