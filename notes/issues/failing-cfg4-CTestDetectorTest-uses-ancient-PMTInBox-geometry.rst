failing-cfg4-CTestDetectorTest-uses-ancient-PMTInBox-geometry
================================================================

* using ancient PmtInBox : not surprising it FAILS




::

    2017-10-18 18:43:48.112 FATAL [197972] [CGeometry::init@59] CGeometry::init G4 simple test geometry 
    2017-10-18 18:43:48.113 INFO  [197972] [GGeo::createSurLib@732] deferred creation of GSurLib 
    2017-10-18 18:43:48.113 INFO  [197972] [GSurLib::collectSur@79]  nsur 48
    2017-10-18 18:43:48.113 INFO  [197972] [CPropLib::init@66] CPropLib::init
    2017-10-18 18:43:48.113 INFO  [197972] [CPropLib::initCheckConstants@118] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-10-18 18:43:48.113 INFO  [197972] [*CTestDetector::makeDetector@120] CTestDetector::makeDetector PmtInBox 1 BoxInBox 0 numSolidsMesh 7 numSolidsConfig 1
    2017-10-18 18:43:48.113 INFO  [197972] [*CTestDetector::makeDetector@163] ni(         0,         0,         0,4294967295)id(         0,         5,         0,         0)
    2017-10-18 18:43:48.113 FATAL [197972] [*CTestDetector::makeDetector@190] CTestDetector::makeDetector changing boundary   0 spec Rock/NONE/perfectAbsorbSurface/MineralOil from boundary0 (from mesh->getNodeInfo()->z ) 0 to boundary (from blib) 123
    2017-10-18 18:43:48.113 INFO  [197972] [GSur::dump@220] isur B(  45                perfectAbsorbSurface)  nlv   0 npvp   0 
    2017-10-18 18:43:48.113 INFO  [197972] [*CTestDetector::makeDetector@211]  spec          Rock/NONE/perfectAbsorbSurface/MineralOil bnd 123 imat 0x107c7af20 isur 0x10c2ae060 osur        0x0
    2017-10-18 18:43:48.114 INFO  [197972] [GCSG::dump@151] CTestDetector::makePMT

    ...


    2017-10-18 18:43:48.115 INFO  [197972] [CTraverser::Traverse@128] CTraverser::Traverse DONE
    2017-10-18 18:43:48.115 INFO  [197972] [CTraverser::Summary@104] CDetector::traverse numMaterials 5 numMaterialsWithoutMPT 0
    2017-10-18 18:43:48.115 INFO  [197972] [CDetector::attachSurfaces@240] CDetector::attachSurfaces
    2017-10-18 18:43:48.116 INFO  [197972] [GSurLib::examineSolidBndSurfaces@115] GSurLib::examineSolidBndSurfaces numSolids 7
    2017-10-18 18:43:48.116 FATAL [197972] [GSurLib::examineSolidBndSurfaces@137] GSurLib::examineSolidBndSurfaces i(mm-idx)      6 node(ni.z)      0 node2(id.x)      0 boundary(id.z)      0 parent(ni.w) 4294967295 bname Vacuum///Vacuum lv __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0
    Assertion failed: (node == i), function examineSolidBndSurfaces, file /Users/blyth/opticks/ggeo/GSurLib.cc, line 147.
    Process 55239 stopped
    * thread #1: tid = 0x30554, 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8b576866:  jae    0x7fff8b576870            ; __pthread_kill + 20
       0x7fff8b576868:  movq   %rax, %rdi
       0x7fff8b57686b:  jmp    0x7fff8b573175            ; cerror_nocancel
       0x7fff8b576870:  retq   
    (lldb) bt
    * thread #1: tid = 0x30554, 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff82c1335c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff89963b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8992d9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000100dee9bd libGGeo.dylib`GSurLib::examineSolidBndSurfaces(this=0x000000010c2abbc0) + 2109 at GSurLib.cc:147
        frame #5: 0x0000000100dee16d libGGeo.dylib`GSurLib::close(this=0x000000010c2abbc0) + 29 at GSurLib.cc:93
        frame #6: 0x00000001016e0c97 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010c2abc20) + 247 at CDetector.cc:244
        frame #7: 0x000000010165b214 libcfg4.dylib`CGeometry::init(this=0x000000010c2ab950) + 1476 at CGeometry.cc:73
        frame #8: 0x000000010165ac40 libcfg4.dylib`CGeometry::CGeometry(this=0x000000010c2ab950, hub=0x00007fff5fbfea40) + 112 at CGeometry.cc:39
        frame #9: 0x000000010165b27d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010c2ab950, hub=0x00007fff5fbfea40) + 29 at CGeometry.cc:40
        frame #10: 0x0000000101703716 libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe988, hub=0x00007fff5fbfea40) + 214 at CG4.cc:122
        frame #11: 0x0000000101703cad libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe988, hub=0x00007fff5fbfea40) + 29 at CG4.cc:144
        frame #12: 0x000000010000d03e CTestDetectorTest`main(argc=1, argv=0x00007fff5fbfede0) + 974 at CTestDetectorTest.cc:50
        frame #13: 0x00007fff869e95fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 12
    frame #12: 0x000000010000d03e CTestDetectorTest`main(argc=1, argv=0x00007fff5fbfede0) + 974 at CTestDetectorTest.cc:50
       47   
       48       OpticksHub hub(&ok);
       49   
    -> 50       CG4 g4(&hub);
       51   
       52       LOG(info) << "CG4 DONE" ; 
       53       CDetector* detector  = g4.getDetector();

    (lldb) f 10
    frame #10: 0x0000000101703716 libcfg4.dylib`CG4::CG4(this=0x00007fff5fbfe988, hub=0x00007fff5fbfea40) + 214 at CG4.cc:122
       119       m_run(m_ok->getRun()),
       120       m_cfg(m_ok->getCfg()),
       121       m_physics(new CPhysics(this)),
    -> 122       m_runManager(m_physics->getRunManager()),
       123       m_geometry(new CGeometry(m_hub)),
       124       m_hookup(m_geometry->hookup(this)),
       125       m_lib(m_geometry->getPropLib()),

    (lldb) f 7
    frame #7: 0x000000010165b214 libcfg4.dylib`CGeometry::init(this=0x000000010c2ab950) + 1476 at CGeometry.cc:73
       70           detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query)) ; 
       71       }
       72   
    -> 73       detector->attachSurfaces();
       74       //m_csurlib->convert(detector);
       75   
       76       m_detector = detector ; 

    (lldb) f 6
    frame #6: 0x00000001016e0c97 libcfg4.dylib`CDetector::attachSurfaces(this=0x000000010c2abc20) + 247 at CDetector.cc:244
       241  
       242  
       243      
    -> 244      m_gsurlib->close();
       245   
       246      m_csurlib = new CSurLib(m_gsurlib);
       247  
    (lldb) 

    (lldb) f 5
    frame #5: 0x0000000100dee16d libGGeo.dylib`GSurLib::close(this=0x000000010c2abbc0) + 29 at GSurLib.cc:93
       90   void GSurLib::close()
       91   {
       92       m_closed = true ; 
    -> 93       examineSolidBndSurfaces();  
       94       assignType();
       95   }




::

    104 void GSurLib::examineSolidBndSurfaces()
    105 {
    106     // this is deferred to CDetector::attachSurfaces 
    107     // to allow CTestDetector to fixup mesh0 info 
    108 
    109     GGeo* gg = m_ggeo ;
    110 
    111     GMergedMesh* mm = gg->getMergedMesh(0) ;
    112 
    113     unsigned numSolids = mm->getNumSolids();
    114 
    115     LOG(info) << "GSurLib::examineSolidBndSurfaces"
    116               << " numSolids " << numSolids
    117               ;
    118 
    119     for(unsigned i=0 ; i < numSolids ; i++)
    120     {
    121         guint4 id = mm->getIdentity(i);
    122         guint4 ni = mm->getNodeInfo(i);
    123         const char* lv = gg->getLVName(i) ;
    124 
    125         // hmm for test geometry the lv returned are the global ones, not the test geometry ones
    126         // and the boundary names look wrong too
    127 
    128         unsigned node = ni.z ;
    129         unsigned parent = ni.w ;
    130 
    131         unsigned node2 = id.x ;
    132         unsigned boundary = id.z ;
    133 
    134         std::string bname = m_blib->shortname(boundary);
    135 
    136         if(node != i)
    137            LOG(fatal) << "GSurLib::examineSolidBndSurfaces"
    138                       << " i(mm-idx) " << std::setw(6) << i
    139                       << " node(ni.z) " << std::setw(6) << node
    140                       << " node2(id.x) " << std::setw(6) << node2
    141                       << " boundary(id.z) " << std::setw(6) << boundary
    142                       << " parent(ni.w) " << std::setw(6) << parent
    143                       << " bname " << bname
    144                       << " lv " << ( lv ? lv : "NULL" )
    145                       ;
    146 


    2017-10-18 18:43:48.115 INFO  [197972] [CDetector::attachSurfaces@240] CDetector::attachSurfaces
    2017-10-18 18:43:48.116 INFO  [197972] [GSurLib::examineSolidBndSurfaces@115] GSurLib::examineSolidBndSurfaces numSolids 7
    2017-10-18 18:43:48.116 FATAL [197972] [GSurLib::examineSolidBndSurfaces@137] GSurLib::examineSolidBndSurfaces 

        i(mm-idx)       6 
        node(ni.z)      0 
        node2(id.x)     0 
        boundary(id.z)  0 
        parent(ni.w)    4294967295 
        bname           Vacuum///Vacuum
        lv              __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0

    Assertion failed: (node == i), function examineSolidBndSurfaces, file /Users/blyth/opticks/ggeo/GSurLib.cc, line 147.



    147         assert( node == i );
    148 
    149 
    150         //unsigned mesh = id.y ;
    151         //unsigned sensor = id.w ;
    152         assert( node2 == i );
    153 
    154         guint4 bnd = m_blib->getBnd(boundary);
    155 
    156         //unsigned omat_ = bnd.x ; 
    157         unsigned osur_ = bnd.y ;
    158         unsigned isur_ = bnd.z ;
    159         //unsigned imat_ = bnd.w ; 





