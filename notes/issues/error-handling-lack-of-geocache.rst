error-handling-lack-of-geocache
=================================

ISSUE
------

Running the *opticks-t* units tests on a new Opticks installation
without the geocache having been built results in lots of 
test failures. Get lots of segv from attempts to load resources
that are not there.

Is it possible to improve the error handling in this
case to give an informative message ?


SIDE ISSUE
------------

VizTest with missing geocache is hanging and forcing a reboot.


REPRODUCE
----------

Setting OPTICKS_QUERY envvar to one that has not been geocached previously (corresponding to a different geo selection)
changes the geocache digest string, so it is a way of mimicking not having a geocache. 
Note that this trick only works once as some of the higher level tests will detect the 
lack of geocache and recreate it.


::

    simon:optickscore blyth$ OPTICKS_QUERY="range:0:1" GMaterialLibTest 
      0 : GMaterialLibTest
    2017-11-27 11:02:16.249 INFO  [7207190] [main@109]  ok 
    2017-11-27 11:02:16.253 WARN  [7207190] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/GItemList/GMaterialLib.txt
    2017-11-27 11:02:16.253 WARN  [7207190] [GMaterialLib::import@455] GMaterialLib::import NULL buffer 
    2017-11-27 11:02:16.253 INFO  [7207190] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    Segmentation fault: 11
    simon:optickscore blyth$ 



::

    simon:optickscore blyth$ OPTICKS_QUERY="range:0:1" lldb GMaterialLibTest 
    (lldb) target create "GMaterialLibTest"
    Current executable set to 'GMaterialLibTest' (x86_64).
    (lldb) r
    Process 62430 launched: '/usr/local/opticks/lib/GMaterialLibTest' (x86_64)
      0 : GMaterialLibTest
    2017-11-27 11:09:57.046 INFO  [7209631] [main@109]  ok 
    2017-11-27 11:09:57.049 WARN  [7209631] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/GItemList/GMaterialLib.txt
    2017-11-27 11:09:57.049 WARN  [7209631] [GMaterialLib::import@455] GMaterialLib::import NULL buffer 
    2017-11-27 11:09:57.049 INFO  [7209631] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    Process 62430 stopped
    * thread #1: tid = 0x6e029f, 0x00000001006ac856 libNPY.dylib`NPYBase::getShape(unsigned int) const [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070, this=0x0000000101400288, __n=140734799798288) const at vector:656, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
        frame #0: 0x00000001006ac856 libNPY.dylib`NPYBase::getShape(unsigned int) const [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070, this=0x0000000101400288, __n=140734799798288) const at vector:656
       653  
       654      _LIBCPP_INLINE_VISIBILITY
       655      size_type size() const _NOEXCEPT
    -> 656          {return static_cast<size_type>(this->__end_ - this->__begin_);}
       657      _LIBCPP_INLINE_VISIBILITY
       658      size_type capacity() const _NOEXCEPT
       659          {return __base::capacity();}
    (lldb) bt
    * thread #1: tid = 0x6e029f, 0x00000001006ac856 libNPY.dylib`NPYBase::getShape(unsigned int) const [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070, this=0x0000000101400288, __n=140734799798288) const at vector:656, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x00000001006ac856 libNPY.dylib`NPYBase::getShape(unsigned int) const [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070, this=0x0000000101400288, __n=140734799798288) const at vector:656
        frame #1: 0x00000001006ac856 libNPY.dylib`NPYBase::getShape(this=0x0000000000000000, n=0) const + 38 at NPYBase.cpp:222
        frame #2: 0x00000001012f6aa3 libGGeo.dylib`GMaterialLib::replaceGROUPVEL(this=0x00000001032049a0, debug=false) + 51 at GMaterialLib.cc:559
        frame #3: 0x00000001012f63ce libGGeo.dylib`GMaterialLib::postLoadFromCache(this=0x00000001032049a0) + 2366 at GMaterialLib.cc:121
        frame #4: 0x00000001012f5a26 libGGeo.dylib`GMaterialLib::load(ok=0x00007fff5fbfebb0) + 86 at GMaterialLib.cc:49
        frame #5: 0x0000000100004a93 GMaterialLibTest`main(argc=1, argv=0x00007fff5fbfed90) + 851 at GMaterialLibTest.cc:111
        frame #6: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 



Reproduce this situation::

    ggeo-nocache-test()
    {
        local dir=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/
        [ -d "$dir" ] && rm -rf $dir 

        OPTICKS_QUERY="range:0:1" lldb GMaterialLibTest 
    }


Improved error handling::

    simon:issues blyth$ ggeo-nocache-test 
    (lldb) target create "GMaterialLibTest"
    Current executable set to 'GMaterialLibTest' (x86_64).
    (lldb) r
    Process 8178 launched: '/usr/local/opticks/lib/GMaterialLibTest' (x86_64)
      0 : GMaterialLibTest
    2017-11-27 11:54:11.929 INFO  [23674] [main@109]  ok 
    2017-11-27 11:54:11.932 FATAL [23674] [GPropertyLib::loadFromCache@500] GPropertyLib::loadFromCache FAILED  dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/GMaterialLib name GMaterialLib.npy YOU PROBABLY NEED TO CREATE THE GEOCACHE BY RUNNING  : op.sh -G 
    Assertion failed: (buf), function loadFromCache, file /Users/blyth/opticks/ggeo/GPropertyLib.cc, line 506.
    Process 8178 stopped
    * thread #1: tid = 0x5c7a, 0x00007fff91d0f866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff91d0f866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff91d0f866:  jae    0x7fff91d0f870            ; __pthread_kill + 20
       0x7fff91d0f868:  movq   %rax, %rdi
       0x7fff91d0f86b:  jmp    0x7fff91d0c175            ; cerror_nocancel
       0x7fff91d0f870:  retq   
    (lldb) 





The just created geocache::


    simon:issues blyth$ ll /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/
    total 0
    drwxr-xr-x    4 blyth  staff   136 Nov 27 11:11 MeshIndex
    drwxr-xr-x    5 blyth  staff   170 Nov 27 11:11 GSurfaceLib
    drwxr-xr-x    3 blyth  staff   102 Nov 27 11:11 GSourceLib
    drwxr-xr-x    5 blyth  staff   170 Nov 27 11:11 GScintillatorLib
    drwxr-xr-x    5 blyth  staff   170 Nov 27 11:11 GNodeLib
    drwxr-xr-x  251 blyth  staff  8534 Nov 27 11:11 GMeshLib
    drwxr-xr-x    8 blyth  staff   272 Nov 27 11:11 GMergedMesh
    drwxr-xr-x    3 blyth  staff   102 Nov 27 11:11 GMaterialLib
    drwxr-xr-x    6 blyth  staff   204 Nov 27 11:11 GItemList
    drwxr-xr-x    3 blyth  staff   102 Nov 27 11:11 GBndLib
    drwxr-xr-x   36 blyth  staff  1224 Nov 27 11:11 ..
    drwxr-xr-x    2 blyth  staff    68 Nov 27 11:15 MeshIndexAnalytic
    drwxr-xr-x   13 blyth  staff   442 Nov 27 11:15 .
    simon:issues blyth$ 



opticks-t without geocache
-----------------------------

::

     rm -rf /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/
     OPTICKS_QUERY="range:0:1" opticks-t


Excluding ok-/VizTest which is hanging with incomplete geocache::

    95% tests passed, 15 tests failed out of 283

    Total Test time (real) =  99.98 sec

    The following tests FAILED:
        177 - GGeoTest.GMaterialLibTest (SEGFAULT)
        180 - GGeoTest.GScintillatorLibTest (SEGFAULT)
        183 - GGeoTest.GBndLibTest (SEGFAULT)
        184 - GGeoTest.GBndLibInitTest (SEGFAULT)
        195 - GGeoTest.GPartsTest (SEGFAULT)
        197 - GGeoTest.GPmtTest (SEGFAULT)
        198 - GGeoTest.BoundariesNPYTest (SEGFAULT)
        199 - GGeoTest.GAttrSeqTest (SEGFAULT)
        203 - GGeoTest.GGeoLibTest (SEGFAULT)
        204 - GGeoTest.GGeoTest (SEGFAULT)
        205 - GGeoTest.GMakerTest (SEGFAULT)
        214 - GGeoTest.NLookupTest (SEGFAULT)
        215 - GGeoTest.RecordsNPYTest (SEGFAULT)
        216 - GGeoTest.GSceneTest (SEGFAULT)
        217 - GGeoTest.GMeshLibTest (OTHER_FAULT)
    Errors while running CTest
    Mon Nov 27 11:30:42 CST 2017
    opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log


Subsequent running of the GGeoTest pass::

    ggeo-;OPTICKS_QUERY="range:0:1" ggeo-t


Fails from Ryan
-----------------

::

    83% tests passed, 45 tests failed out of 267

    Total Test time (real) =  79.21 sec

    The following tests FAILED:
          5 - SysRapTest.SSysTest (Failed)     

              ## explained, missing dir in PATH

        177 - GGeoTest.GMaterialLibTest (SEGFAULT)
         80 - GGeoTest.GScintillatorLibTest (SEGFAULT)
        183 - GGeoTest.GBndLibTest (SEGFAULT)
        184 - GGeoTest.GBndLibInitTest (SEGFAULT)
        195 - GGeoTest.GPartsTest (SEGFAULT)
        197 - GGeoTest.GPmtTest (SEGFAULT)
        198 - GGeoTest.BoundariesNPYTest (SEGFAULT)
        199 - GGeoTest.GAttrSeqTest (SEGFAULT)
        203 - GGeoTest.GGeoLibTest (SEGFAULT)
        204 - GGeoTest.GGeoTest (SEGFAULT)
        205 - GGeoTest.GMakerTest (SEGFAULT)
        214 - GGeoTest.NLookupTest (SEGFAULT)
        215 - GGeoTest.RecordsNPYTest (SEGFAULT)
        216 - GGeoTest.GSceneTest (SEGFAULT)
        217 - GGeoTest.GMeshLibTest (OTHER_FAULT)   

              ## reproduced above should be fixed by:  op.sh -G 
              ## have improved error handling of these

        218 - AssimpRapTest.AssimpRapTest (SEGFAULT)
        220 - AssimpRapTest.AssimpGGeoTest (SEGFAULT)
        222 - OpticksGeometryTest.OpticksGeometryTest (SEGFAULT)
        223 - OpticksGeometryTest.OpticksHubTest (SEGFAULT)
        224 - OpticksGeometryTest.OpenMeshRapTest (SEGFAULT)
        241 - OptiXRapTest.OScintillatorLibTest (SEGFAULT)
        242 - OptiXRapTest.OOTextureTest (SEGFAULT)
        243 - OptiXRapTest.OOMinimalTest (OTHER_FAULT)
        244 - OptiXRapTest.OOContextTest (OTHER_FAULT)
        245 - OptiXRapTest.OOContextUploadDownloadTest (OTHER_FAULT)
        246 - OptiXRapTest.LTOOContextUploadDownloadTest (OTHER_FAULT)
        247 - OptiXRapTest.OOboundaryTest (SEGFAULT)
        248 - OptiXRapTest.OOboundaryLookupTest (SEGFAULT)
        249 - OptiXRapTest.OOtex0Test (OTHER_FAULT)
        250 - OptiXRapTest.OOtexTest (OTHER_FAULT)
        251 - OptiXRapTest.bufferTest (OTHER_FAULT)
        252 - OptiXRapTest.OEventTest (SEGFAULT)
        253 - OptiXRapTest.OInterpolationTest (SEGFAULT)
        254 - OptiXRapTest.ORayleighTest (SEGFAULT)
        255 - OptiXRapTest.intersect_analytic_test (OTHER_FAULT)
        256 - OptiXRapTest.Roots3And4Test (OTHER_FAULT)
        257 - OKOPTest.OpIndexerTest (SEGFAULT)
        258 - OKOPTest.OpSeederTest (SEGFAULT)
        259 - OKOPTest.dirtyBufferTest (OTHER_FAULT)
        260 - OKOPTest.compactionTest (OTHER_FAULT)
        261 - OKOPTest.OpTest (SEGFAULT)
        263 - OKTest.OKTest (SEGFAULT)
        264 - OKTest.OTracerTest (SEGFAULT)

              ## not yet investigated
            

        266 - OKTest.VizTest (SEGFAULT)            ## now excluded as was causing an unexplained hang, forcing reboot 
    Errors while running CTest
    Sun Nov 26 17:35:45 EST 2017






