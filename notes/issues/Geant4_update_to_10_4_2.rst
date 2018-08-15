Geant4_update_to_10_4_2
=========================

* https://bitbucket.org/simoncblyth/opticks/commits/db6a93f072ed6ce5ef98556f5c2f1884d1de7c6f


* theParticleIterator
* needed to const_cast G4MaterialPropertiesTable
* getting NULL dynamicparticle

* X4PhysicsVector::Digest reproducibly SIGABRT on macOS, workaround get digest from converted GProperty 


ckm- polygonize crash, and dont have debug symbols ?? for G4 

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x48)
      * frame #0: 0x00000001029d02b9 libG4geometry.dylib`G4Box::GetExtent() const + 9
        frame #1: 0x000000010066ad2d libExtG4.dylib`X4Mesh::polygonize(this=0x00007ffeefbfc7c0) at X4Mesh.cc:128
        frame #2: 0x0000000100669fbf libExtG4.dylib`X4Mesh::init(this=0x00007ffeefbfc7c0) at X4Mesh.cc:93
        frame #3: 0x0000000100669f92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:83
        frame #4: 0x0000000100669f0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x00007ffeefbfc7c0, solid=0x000000010e297bd0) at X4Mesh.cc:82
        frame #5: 0x0000000100669ebc libExtG4.dylib`X4Mesh::Convert(solid=0x000000010e297bd0) at X4Mesh.cc:67

X4MeshTest has the same issue with GetExtent()::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x0000000102491f09 libG4geometry.dylib`G4Sphere::GetExtent() const + 9
        frame #1: 0x000000010010dd2d libExtG4.dylib`X4Mesh::polygonize(this=0x0000000106814740) at X4Mesh.cc:128
        frame #2: 0x000000010010cfbf libExtG4.dylib`X4Mesh::init(this=0x0000000106814740) at X4Mesh.cc:93
        frame #3: 0x000000010010cf92 libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:83
        frame #4: 0x000000010010cf0d libExtG4.dylib`X4Mesh::X4Mesh(this=0x0000000106814740, solid=0x00000001068147d0) at X4Mesh.cc:82
        frame #5: 0x000000010000da58 X4MeshTest`main(argc=2, argv=0x00007ffeefbfea88) at X4MeshTest.cc:16
        frame #6: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #7: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 


Nuclear option as g4-configure omitted to setup Debug build.::

   g4-wipe
   g4--



After replacing all use of G4VisExtent for X4SolidExtent all X4 tests pass::

    epsilon:extg4 blyth$ om-test
    === om-test-one : extg4           /Users/blyth/opticks/extg4                                   /usr/local/opticks/build/extg4                               
    Wed Aug 15 23:16:39 CST 2018
    Test project /usr/local/opticks/build/extg4
          Start  1: ExtG4Test.X4Test
     1/17 Test  #1: ExtG4Test.X4Test ..................   Passed    0.04 sec
          Start  2: ExtG4Test.X4EntityTest
     2/17 Test  #2: ExtG4Test.X4EntityTest ............   Passed    0.04 sec
          Start  3: ExtG4Test.X4SolidTest
     3/17 Test  #3: ExtG4Test.X4SolidTest .............   Passed    0.20 sec
          Start  4: ExtG4Test.X4SolidLoadTest
     4/17 Test  #4: ExtG4Test.X4SolidLoadTest .........   Passed    0.04 sec
          Start  5: ExtG4Test.X4MeshTest
     5/17 Test  #5: ExtG4Test.X4MeshTest ..............   Passed    0.04 sec
          Start  6: ExtG4Test.X4SolidExtentTest
     6/17 Test  #6: ExtG4Test.X4SolidExtentTest .......   Passed    0.04 sec
          Start  7: ExtG4Test.X4SolidListTest
     7/17 Test  #7: ExtG4Test.X4SolidListTest .........   Passed    0.04 sec
          Start  8: ExtG4Test.X4PhysicsVectorTest
     8/17 Test  #8: ExtG4Test.X4PhysicsVectorTest .....   Passed    0.04 sec
          Start  9: ExtG4Test.X4MaterialTest
     9/17 Test  #9: ExtG4Test.X4MaterialTest ..........   Passed    0.05 sec
          Start 10: ExtG4Test.X4MaterialTableTest
    10/17 Test #10: ExtG4Test.X4MaterialTableTest .....   Passed    0.05 sec
          Start 11: ExtG4Test.X4PhysicalVolumeTest
    11/17 Test #11: ExtG4Test.X4PhysicalVolumeTest ....   Passed    0.06 sec
          Start 12: ExtG4Test.X4PhysicalVolume2Test
    12/17 Test #12: ExtG4Test.X4PhysicalVolume2Test ...   Passed    0.06 sec
          Start 13: ExtG4Test.X4Transform3DTest
    13/17 Test #13: ExtG4Test.X4Transform3DTest .......   Passed    0.04 sec
          Start 14: ExtG4Test.X4AffineTransformTest
    14/17 Test #14: ExtG4Test.X4AffineTransformTest ...   Passed    0.04 sec
          Start 15: ExtG4Test.X4ThreeVectorTest
    15/17 Test #15: ExtG4Test.X4ThreeVectorTest .......   Passed    0.04 sec
          Start 16: ExtG4Test.X4CSGTest
    16/17 Test #16: ExtG4Test.X4CSGTest ...............   Passed    0.04 sec
          Start 17: ExtG4Test.X4PolyconeTest
    17/17 Test #17: ExtG4Test.X4PolyconeTest ..........   Passed    0.04 sec

    100% tests passed, 0 tests failed out of 17

    Total Test time (real) =   0.95 sec
    Wed Aug 15 23:16:40 CST 2018
    epsilon:extg4 blyth$ 





CFG4 still has 6 fails
------------------------

::

    74% tests passed, 6 tests failed out of 23

    Total Test time (real) =   7.46 sec

    The following tests FAILED:
          3 - CFG4Test.CTestDetectorTest (SEGFAULT)
          4 - CFG4Test.CGDMLDetectorTest (Child aborted)
          5 - CFG4Test.CGeometryTest (Child aborted)
          6 - CFG4Test.CG4Test (SEGFAULT)
         18 - CFG4Test.CInterpolationTest (SEGFAULT)
         23 - CFG4Test.CRandomEngineTest (SEGFAULT)
    Errors while running CTest
    Wed Aug 15 23:06:55 CST 2018
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 


ENSDFSTATE issue
-------------------

The 4 SEGFAULT are all from the same cause::

    epsilon:tests blyth$ CRandomEngineTest
    2018-08-15 23:13:51.740 INFO  [1395722] [main@72] CRandomEngineTest
    2018-08-15 23:13:51.740 INFO  [1395722] [main@76]  pindex 0
      0 : CRandomEngineTest

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70002
          issued by : G4NuclideTable
    5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.
    *** Fatal Exception *** core dump ***
    Segmentation fault: 11
    epsilon:tests blyth$ 
    epsilon:tests blyth$ echo $ENSDFSTATE

    epsilon:tests blyth$ t g4-export
    g4-export is a function
    g4-export () 
    { 
        source $(g4-sh)
    }
    epsilon:tests blyth$ g4-sh
    /usr/local/opticks/externals/bin/geant4.sh
    epsilon:tests blyth$ vi /usr/local/opticks/externals/bin/geant4.sh
    epsilon:tests blyth$ env | grep G4
    G4LEVELGAMMADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4NDL4.5
    G4ENSDFSTATEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4RADIOACTIVEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    G4ABLADATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ABLA3.1
    G4PIIDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4PII1.3
    G4SAIDXSDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/usr/local/opticks/externals/share/Geant4-10.4.2/data/RealSurface2.1.1



::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=EXC_I386_GPFLT)
      * frame #0: 0x00000001007e917c libG4vis_management.dylib`G4EventManager::GetTrackingManager(this=0x4e706574736e6547) const at G4EventManager.hh:165
        frame #1: 0x00000001007df339 libG4vis_management.dylib`G4RunManagerKernel::GetTrackingManager(this=0x000000010e7574d0) const at G4RunManagerKernel.hh:183
        frame #2: 0x0000000101e28ca2 libG4run.dylib`G4ExceptionHandler::DumpTrackInfo(this=0x000000010e7572f0) at G4ExceptionHandler.cc:151
        frame #3: 0x0000000101e283f4 libG4run.dylib`G4ExceptionHandler::Notify(this=0x000000010e7572f0, originOfException="G4NuclideTable", exceptionCode="PART70002", severity=FatalException, description="5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.") at G4ExceptionHandler.cc:95
        frame #4: 0x0000000105b662fb libG4global.dylib`G4Exception(originOfException="G4NuclideTable", exceptionCode="PART70002", severity=FatalException, description="5.609049e+17 is not valid indicator of G4Ions::G4FloatLevelBase. You may use a wrong version of ENSDFSTATE data. Please use G4ENSDFSTATE2.0 or later.") at G4Exception.cc:52
        frame #5: 0x000000010516d4e3 libG4particles.dylib`G4NuclideTable::StripFloatLevelBase(this=0x0000000105246e70, sFLB=(std::__1::string = "5.609049e+17")) at G4NuclideTable.cc:395
        frame #6: 0x00000001051679e5 libG4particles.dylib`G4NuclideTable::GenerateNuclide(this=0x0000000105246e70) at G4NuclideTable.cc:228
        frame #7: 0x00000001051672f8 libG4particles.dylib`G4NuclideTable::G4NuclideTable(this=0x0000000105246e70) at G4NuclideTable.cc:74
        frame #8: 0x0000000105167085 libG4particles.dylib`G4NuclideTable::G4NuclideTable(this=0x0000000105246e70) at G4NuclideTable.cc:69
        frame #9: 0x0000000105167006 libG4particles.dylib`G4NuclideTable::GetInstance() at G4NuclideTable.cc:57
        frame #10: 0x000000010513d559 libG4particles.dylib`G4NuclideTable::GetNuclideTable() at G4NuclideTable.hh:73
        frame #11: 0x000000010513c484 libG4particles.dylib`G4IonTable::PrepareNuclideTable(this=0x000000010e757580) at G4IonTable.cc:1666
        frame #12: 0x000000010513c443 libG4particles.dylib`G4IonTable::G4IonTable(this=0x000000010e757580) at G4IonTable.cc:145
        frame #13: 0x000000010513c7e5 libG4particles.dylib`G4IonTable::G4IonTable(this=0x000000010e757580) at G4IonTable.cc:126
        frame #14: 0x0000000105181c79 libG4particles.dylib`G4ParticleTable::G4ParticleTable(this=0x0000000105246f60) at G4ParticleTable.cc:147
        frame #15: 0x00000001051812b5 libG4particles.dylib`G4ParticleTable::G4ParticleTable(this=0x0000000105246f60) at G4ParticleTable.cc:118
        frame #16: 0x00000001051811f6 libG4particles.dylib`G4ParticleTable::GetParticleTable() at G4ParticleTable.cc:99
        frame #17: 0x0000000101e662d6 libG4run.dylib`G4RunManagerKernel::G4RunManagerKernel(this=0x000000010e7574d0) at G4RunManagerKernel.cc:102
        frame #18: 0x0000000101e673e5 libG4run.dylib`G4RunManagerKernel::G4RunManagerKernel(this=0x000000010e7574d0) at G4RunManagerKernel.cc:88
        frame #19: 0x0000000101e44039 libG4run.dylib`G4RunManager::G4RunManager(this=0x000000010e757360) at G4RunManager.cc:105
        frame #20: 0x0000000101e449d5 libG4run.dylib`G4RunManager::G4RunManager(this=0x000000010e757360) at G4RunManager.cc:97
        frame #21: 0x000000010011c2ab libCFG4.dylib`CPhysics::CPhysics(this=0x000000010e757330, g4=0x00007ffeefbfe0f0) at CPhysics.cc:19
        frame #22: 0x000000010011c3ad libCFG4.dylib`CPhysics::CPhysics(this=0x000000010e757330, g4=0x00007ffeefbfe0f0) at CPhysics.cc:25
        frame #23: 0x00000001001e3945 libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe0f0, hub=0x00007ffeefbfe298) at CG4.cc:107
        frame #24: 0x00000001001e428d libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe0f0, hub=0x00007ffeefbfe298) at CG4.cc:128
        frame #25: 0x000000010000eea9 CInterpolationTest`main(argc=1, argv=0x00007ffeefbfea88) at CInterpolationTest.cc:57
        frame #26: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #27: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 


* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/installconfig/1887.html

::

    epsilon:extg4 blyth$ ll /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/
    total 3504
    -rw-r--r--   1 blyth  staff      436 Sep 16  2016 README
    -rw-r--r--   1 blyth  staff  1785840 Sep  5  2017 ENSDFSTATE.dat
    -rw-r--r--   1 blyth  staff     1476 Sep 26  2017 History
    drwxr-xr-x  12 blyth  staff      384 Aug 15 18:40 ..
    drwxr-xr-x   5 blyth  staff      160 Aug 15 18:40 .
    epsilon:extg4 blyth$ ll /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/ENSDFSTATE.dat 
    -rw-r--r--  1 blyth  staff  1785840 Sep  5  2017 /usr/local/opticks/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2/ENSDFSTATE.dat
    epsilon:extg4 blyth$ 







2 SIGABRT : from same assert related to skin surfaces
------------------------------------------------------

::

    018-08-15 23:36:09.167 INFO  [1435854] [CDetector::attachSurfaces@277]  num_bs 0 num_sk 0
    2018-08-15 23:36:09.167 INFO  [1435854] [CDetector::attachSurfaces@289] [--dbgsurf] CDetector::attachSurfaces START
    2018-08-15 23:36:09.167 INFO  [1435854] [CSurfaceLib::convert@81] .
    2018-08-15 23:36:09.167 INFO  [1435854] [CSurfaceLib::convert@93] . num_surf 48
    Assertion failed: (lv), function makeSkinSurface, file /Users/blyth/opticks/cfg4/CSurfaceLib.cc, line 249.
    Process 92845 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53502b6e <+10>: jae    0x7fff53502b78            ; <+20>
        0x7fff53502b70 <+12>: movq   %rax, %rdi
        0x7fff53502b73 <+15>: jmp    0x7fff534f9b00            ; cerror_nocancel
        0x7fff53502b78 <+20>: retq   
    Target 0: (CGDMLDetectorTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff536cd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5345e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff534261ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001ca1d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a732b40, surf=0x000000010a584c60, os=0x000000010ddd9c30) at CSurfaceLib.cc:249
        frame #5: 0x00000001001c8bbb libCFG4.dylib`CSurfaceLib::convert(this=0x000000010a732b40, detector=0x000000010a732a60, exclude_sensors=true) at CSurfaceLib.cc:124
        frame #6: 0x00000001001c149a libCFG4.dylib`CDetector::attachSurfaces(this=0x000000010a732a60) at CDetector.cc:292
        frame #7: 0x00000001001c5ef6 libCFG4.dylib`CGDMLDetector::init(this=0x000000010a732a60) at CGDMLDetector.cc:75
        frame #8: 0x00000001001c5bbb libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a732a60, hub=0x00007ffeefbfe2e0, query=0x000000010b8105b0) at CGDMLDetector.cc:40
        frame #9: 0x00000001001c5f35 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a732a60, hub=0x00007ffeefbfe2e0, query=0x000000010b8105b0) at CGDMLDetector.cc:38
        frame #10: 0x000000010000f5a4 CGDMLDetectorTest`main(argc=1, argv=0x00007ffeefbfe6e0) at CGDMLDetectorTest.cc:51
        frame #11: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x00000001001ca1d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a732b40, surf=0x000000010a584c60, os=0x000000010ddd9c30) at CSurfaceLib.cc:249
       246 	              << " lv " << ( lv ? lv->GetName() : "NULL" )
       247 	              ;
       248 	
    -> 249 	    assert(lv) ;
       250 	
       251 	    G4LogicalSkinSurface* lss = new G4LogicalSkinSurface(name, const_cast<G4LogicalVolume*>(lv), os );
       252 	    return lss ;
    (lldb) p lvn
    (char *) $0 = 0x000000010ddd94d0 "/dd/Geometry/PoolDetails/lvNearTopCover0xc137060"
    (lldb) 
    (lldb) p sslv
    (std::__1::string) $1 = "__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"
    (lldb) p name
    (const char *) $2 = 0x000000010a584c69 "NearPoolCoverSurface"
    (lldb) 

    (lldb) p m_detector->m_traverser->description()
    (std::__1::string) $5 = " numSelected 9068 bbox NBoundingBox low -23327.6914,-809820.6250,-12110.0000 high -9712.3086,-794399.3750,-2140.0000 ce -16520.0000,-802110.0000,-7125.0000,7710.6250 pvs.size 12230 lvs.size 12230"

    (lldb) p m_detector->m_traverser->m_lvm
    (std::__1::map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *, std::__1::less<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<const std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *> > >) $6 = size=1012 {
      [0] = {
        first = ""
        second = 0x00000001128da2d0
      }
      [1] = {
        first = "\x02?
        second = 0x0000000111f01740
      }
      [2] = {
        first = "\x02?
        second = 0x0000000111f01740
      }
      [3] = {
        first = "\x04?
        second = 0x000000010a7fbc50
      }
      [4] = {
        first = "\x04?
        second = 0x000000010a7fbc50
      }
      [5] = {
        first = "\x06\x7f\n\x01\0\0\0A\0\0\0\0\0\0\0:\0\0\0\0\0\0\0\0O?\b%?\a\x02\0\0\0\0\0\0\0\0\0`\x84\x8cP\U0000007f\0\0?F?7?\x7f\0\0\x01ar, std\0\0\0\0\0\0\0\0\0J?7?\x7f\0\0"
        second = 0x000000010a7f0490
      }


Garbled names::

    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn ?
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn (?
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 
    2018-08-15 23:48:01.751 INFO  [1444508] [CTraverser::AncestorVisit@233]  lvn 









    2018-08-15 23:38:23.321 INFO  [1438333] [CSurfaceLib::convert@93] . num_surf 48
    Assertion failed: (lv), function makeSkinSurface, file /Users/blyth/opticks/cfg4/CSurfaceLib.cc, line 249.
    Process 92901 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff53502b6e <+10>: jae    0x7fff53502b78            ; <+20>
        0x7fff53502b70 <+12>: movq   %rax, %rdi
        0x7fff53502b73 <+15>: jmp    0x7fff534f9b00            ; cerror_nocancel
        0x7fff53502b78 <+20>: retq   
    Target 0: (CGeometryTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff53502b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff536cd080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff5345e1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff534261ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001c71d6 libCFG4.dylib`CSurfaceLib::makeSkinSurface(this=0x000000010a79b860, surf=0x000000010a587980, os=0x0000000110a8acb0) at CSurfaceLib.cc:249
        frame #5: 0x00000001001c5bbb libCFG4.dylib`CSurfaceLib::convert(this=0x000000010a79b860, detector=0x000000010a79b6c0, exclude_sensors=true) at CSurfaceLib.cc:124
        frame #6: 0x00000001001be49a libCFG4.dylib`CDetector::attachSurfaces(this=0x000000010a79b6c0) at CDetector.cc:292
        frame #7: 0x00000001001c2ef6 libCFG4.dylib`CGDMLDetector::init(this=0x000000010a79b6c0) at CGDMLDetector.cc:75
        frame #8: 0x00000001001c2bbb libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a79b6c0, hub=0x00007ffeefbfe690, query=0x000000010a512890) at CGDMLDetector.cc:40
        frame #9: 0x00000001001c2f35 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a79b6c0, hub=0x00007ffeefbfe690, query=0x000000010a512890) at CGDMLDetector.cc:38
        frame #10: 0x0000000100119a0a libCFG4.dylib`CGeometry::init(this=0x00007ffeefbfe650) at CGeometry.cc:66
        frame #11: 0x0000000100119730 libCFG4.dylib`CGeometry::CGeometry(this=0x00007ffeefbfe650, hub=0x00007ffeefbfe690) at CGeometry.cc:49
        frame #12: 0x0000000100119a9d libCFG4.dylib`CGeometry::CGeometry(this=0x00007ffeefbfe650, hub=0x00007ffeefbfe690) at CGeometry.cc:48
        frame #13: 0x000000010000f7a5 CGeometryTest`main(argc=1, argv=0x00007ffeefbfea98) at CGeometryTest.cc:45
        frame #14: 0x00007fff533b2015 libdyld.dylib`start + 1
        frame #15: 0x00007fff533b2015 libdyld.dylib`start + 1
    (lldb) 

