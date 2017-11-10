GGeoTest_isClosed_assert
==========================


FIXED ISSUE
-------------

* with GGeoTest something is triggering a close that trips an assert 
  on adding another material

* With full geometry it is OK to add all items before needing to getIndex
  and triggering a close : but with test geometry that aint paractical, unless...

* problem is for full geometry the items have some sort order, 
  so more relevant items can have the lower indices (not just 
  cosmetic, its relevant as use nibble per-step seqmat etc... )


TODO 
------

1. check geometry with multiple different surfaces, suspect
   there will be an analogous issue


Approaches
-----------

1. kludge switch for test geometry to switch off the sorting, so 
   the live as added indices stay correct

2. just mimic the GGeo test spin over materials before over 
   structure... ie collect materials first before doing the boundaries
   in GGeoTest 

**FIXED USING APPROACH 2**



::

    376 void GPropertyLib::close()
    377 {
    378     if(m_ok->isDbgClose())
    379     {
    380         LOG(fatal) << "[--dbgclose] hari-kari " ;
    381         assert(0);
    382     }
    383 
    384     LOG(trace) << "GPropertyLib::close" ;
    385 
    386     sort();
    387 
    388     LOG(trace) << "GPropertyLib::close after sort " ;
    389 
    390     // create methods from sub-class specializations
    391 
    392     GItemList* names = createNames();
    393     NPY<float>* buf = createBuffer() ;
    394     NMeta* meta = createMeta();
    395 
    396     LOG(info) << "GPropertyLib::close"
    397               << " type " << m_type
    398               << " buf " <<  ( buf ? buf->getShapeString() : "NULL" )
    399               ;
    400 
    401     //names->dump("GPropertyLib::close") ;
    402 
    403     setNames(names);
    404     setBuffer(buf);
    405     setMeta(meta);
    406     setClosed();
    407 
    408     LOG(trace) << "GPropertyLib::close DONE" ;
    409 }










::

    tboolean-;tboolean-sphere --okg4 -D

    ...



    2017-11-10 19:57:29.759 INFO  [4302413] [*OpticksHub::getGGeoBasePrimary@687] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 0 ggb GGeo
    2017-11-10 19:57:29.759 INFO  [4302413] [*OpticksHub::createTestGeometry@411] OpticksHub::createTestGeometry START
    2017-11-10 19:57:29.762 FATAL [4302413] [GGeoTest::GGeoTest@107] GGeoTest::GGeoTest
    2017-11-10 19:57:29.762 INFO  [4302413] [*GGeoTest::initCreate@172] GGeoTest::initCreate START  mode PyCsgInBox
    2017-11-10 19:57:29.762 INFO  [4302413] [NTxt::dump@47] NCSGList::load NumLines 2
    Rock//perfectAbsorbSurface/Vacuum
    Vacuum///Pyrex
    2017-11-10 19:57:29.762 INFO  [4302413] [NCSGList::load@82] NCSGList::load VERBOSITY 0 basedir /tmp/blyth/opticks/tboolean-sphere-- txtpath /tmp/blyth/opticks/tboolean-sphere--/csg.txt nbnd 2
    2017-11-10 19:57:29.763 INFO  [4302413] [GGeoTest::loadCSG@312] GGeoTest::loadCSG START  csgpath /tmp/blyth/opticks/tboolean-sphere-- ntree 2 verbosity 0
    2017-11-10 19:57:30.254 INFO  [4302413] [*GMaker::makeFromCSG@182] GMaker::makeFromCSG verbosity 0 index 0 boundary-spec Rock//perfectAbsorbSurface/Vacuum numTris 119996 trisMsg 
    2017-11-10 19:57:30.254 INFO  [4302413] [GPropertyLib::getIndex@341] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Rock]
    2017-11-10 19:57:30.254 INFO  [4302413] [GPropertyLib::close@385] GPropertyLib::close type GMaterialLib buf 2,2,39,4
    2017-11-10 19:57:30.254 INFO  [4302413] [GPropertyLib::getIndex@341] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-11-10 19:57:30.254 INFO  [4302413] [GPropertyLib::close@385] GPropertyLib::close type GSurfaceLib buf 1,2,39,4
    2017-11-10 19:57:30.605 INFO  [4302413] [*GMaker::makeFromCSG@182] GMaker::makeFromCSG verbosity 0 index 1 boundary-spec Vacuum///Pyrex numTris 94316 trisMsg 
    Assertion failed: (!isClosed()), function addDirect, file /Users/blyth/opticks/ggeo/GMaterialLib.cc, line 230.
    Process 69408 stopped
    * thread #1: tid = 0x41a64d, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x41a64d, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000102108806 libGGeo.dylib`GMaterialLib::addDirect(this=0x000000010c5b8a40, mat=0x000000010972f6e0) + 102 at GMaterialLib.cc:230
        frame #5: 0x000000010210cfe0 libGGeo.dylib`GMaterialLib::reuseBasisMaterial(this=0x000000010c5b8a40, name=0x000000010c5da7e9) + 400 at GMaterialLib.cc:789
        frame #6: 0x0000000102179ad5 libGGeo.dylib`GGeoTest::reuseMaterials(this=0x000000010c5b8750, spec=0x000000010c5bc660) + 293 at GGeoTest.cc:290
        frame #7: 0x0000000102179968 libGGeo.dylib`GGeoTest::boundarySetup(this=0x000000010c5b8750, solid=0x000000010c5da090, spec=0x000000010c5bc660) + 40 at GGeoTest.cc:230
        frame #8: 0x0000000102178624 libGGeo.dylib`GGeoTest::loadCSG(this=0x000000010c5b8750, csgpath=0x000000010c5b89e0, solids=0x000000010c5c3910) + 1588 at GGeoTest.cc:347
        frame #9: 0x0000000102177c16 libGGeo.dylib`GGeoTest::initCreate(this=0x000000010c5b8750) + 1078 at GGeoTest.cc:183
        frame #10: 0x000000010217747a libGGeo.dylib`GGeoTest::init(this=0x000000010c5b8750) + 218 at GGeoTest.cc:125
        frame #11: 0x000000010217738a libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010c5b8750, ok=0x0000000109631530, basis=0x0000000109713810) + 1562 at GGeoTest.cc:109
        frame #12: 0x00000001021776d5 libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010c5b8750, ok=0x0000000109631530, basis=0x0000000109713810) + 37 at GGeoTest.cc:110
        frame #13: 0x000000010230e665 libOpticksGeometry.dylib`OpticksHub::createTestGeometry(this=0x000000010970ed30, basis=0x0000000109713810) + 357 at OpticksHub.cc:413
        frame #14: 0x000000010230d4cc libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x000000010970ed30) + 844 at OpticksHub.cc:389
        frame #15: 0x000000010230c359 libOpticksGeometry.dylib`OpticksHub::init(this=0x000000010970ed30) + 137 at OpticksHub.cc:182
        frame #16: 0x000000010230c226 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010970ed30, ok=0x0000000109631530) + 454 at OpticksHub.cc:166
        frame #17: 0x000000010230c45d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010970ed30, ok=0x0000000109631530) + 29 at OpticksHub.cc:168
        frame #18: 0x00000001044d8bbb libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe460, argc=27, argv=0x00007fff5fbfe540) + 283 at OKG4Mgr.cc:30
        frame #19: 0x00000001044d8f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe460, argc=27, argv=0x00007fff5fbfe540) + 35 at OKG4Mgr.cc:41
        frame #20: 0x00000001000132ee OKG4Test`main(argc=27, argv=0x00007fff5fbfe540) + 1486 at OKG4Test.cc:56
        frame #21: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 


Original close::

    tboolean-;tboolean-sphere --okg4 -D --dbgclose


    2017-11-10 20:14:00.393 INFO  [4311907] [GPropertyLib::getIndex@341] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Rock]
    2017-11-10 20:14:00.393 FATAL [4311907] [GPropertyLib::close@376] [--dbgclose] hari-kari 
    Assertion failed: (0), function close, file /Users/blyth/opticks/ggeo/GPropertyLib.cc, line 377.
    Process 73585 stopped


    (lldb) f 6
    frame #6: 0x0000000102132938 libGGeo.dylib`GBnd::init(this=0x00007fff5fbfc5b8, flip_=false) + 3368 at GBnd.cc:51
       48           imat_ = elem[0].c_str() ; 
       49       }
       50   
    -> 51       omat = mlib->getIndex(omat_) ;
       52       osur = slib->getIndex(osur_) ;
       53       isur = slib->getIndex(isur_) ;
       54       imat = mlib->getIndex(imat_) ;
    (lldb) p omat_
    (const char *) $1 = 0x000000010c5d0581 "Rock"
    (lldb) bt
    * thread #1: tid = 0x41cb63, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001020fe9cf libGGeo.dylib`GPropertyLib::close(this=0x000000010c5b8c20) + 287 at GPropertyLib.cc:377
        frame #5: 0x00000001020fe828 libGGeo.dylib`GPropertyLib::getIndex(this=0x000000010c5b8c20, shortname=0x000000010c5d0581) + 488 at GPropertyLib.cc:348
      * frame #6: 0x0000000102132938 libGGeo.dylib`GBnd::init(this=0x00007fff5fbfc5b8, flip_=false) + 3368 at GBnd.cc:51
        frame #7: 0x0000000102131bc6 libGGeo.dylib`GBnd::GBnd(this=0x00007fff5fbfc5b8, spec_=0x000000010c5c5a60, flip_=false, mlib_=0x000000010c5b8c20, slib_=0x000000010c5bbab0, dbgbnd_=false) + 294 at GBnd.cc:16
        frame #8: 0x0000000102132e74 libGGeo.dylib`GBnd::GBnd(this=0x00007fff5fbfc5b8, spec_=0x000000010c5c5a60, flip_=false, mlib_=0x000000010c5b8c20, slib_=0x000000010c5bbab0, dbgbnd_=false) + 84 at GBnd.cc:18
        frame #9: 0x0000000102135a5a libGGeo.dylib`GBndLib::addBoundary(this=0x000000010c5c3190, spec=0x000000010c5c5a60, flip=false) + 122 at GBndLib.cc:322
        frame #10: 0x000000010217994f libGGeo.dylib`GGeoTest::boundarySetup(this=0x000000010c5b8930, solid=0x000000010c5cfcb0, spec=0x000000010c5c5a60) + 79 at GGeoTest.cc:233
        frame #11: 0x00000001021785e4 libGGeo.dylib`GGeoTest::loadCSG(this=0x000000010c5b8930, csgpath=0x000000010c5b8bc0, solids=0x000000010c5c3af0) + 1588 at GGeoTest.cc:347
        frame #12: 0x0000000102177bd6 libGGeo.dylib`GGeoTest::initCreate(this=0x000000010c5b8930) + 1078 at GGeoTest.cc:183
        frame #13: 0x000000010217743a libGGeo.dylib`GGeoTest::init(this=0x000000010c5b8930) + 218 at GGeoTest.cc:125
        frame #14: 0x000000010217734a libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010c5b8930, ok=0x0000000109631560, basis=0x0000000109715f70) + 1562 at GGeoTest.cc:109
        frame #15: 0x0000000102177695 libGGeo.dylib`GGeoTest::GGeoTest(this=0x000000010c5b8930, ok=0x0000000109631560, basis=0x0000000109715f70) + 37 at GGeoTest.cc:110
        frame #16: 0x000000010230e665 libOpticksGeometry.dylib`OpticksHub::createTestGeometry(this=0x000000010970ede0, basis=0x0000000109715f70) + 357 at OpticksHub.cc:413
        frame #17: 0x000000010230d4cc libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x000000010970ede0) + 844 at OpticksHub.cc:389
        frame #18: 0x000000010230c359 libOpticksGeometry.dylib`OpticksHub::init(this=0x000000010970ede0) + 137 at OpticksHub.cc:182
        frame #19: 0x000000010230c226 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010970ede0, ok=0x0000000109631560) + 454 at OpticksHub.cc:166
        frame #20: 0x000000010230c45d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010970ede0, ok=0x0000000109631560) + 29 at OpticksHub.cc:168
        frame #21: 0x00000001044d8bbb libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe440, argc=28, argv=0x00007fff5fbfe520) + 283 at OKG4Mgr.cc:30
        frame #22: 0x00000001044d8f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe440, argc=28, argv=0x00007fff5fbfe520) + 35 at OKG4Mgr.cc:41
        frame #23: 0x00000001000132ee OKG4Test`main(argc=28, argv=0x00007fff5fbfe520) + 1486 at OKG4Test.cc:56
        frame #24: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 




