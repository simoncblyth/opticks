tboolean-bib-source(){   echo $(opticks-home)/tests/tboolean-bib.bash ; }
tboolean-bib-vi(){       vi $(tboolean-bib-source) ; }
tboolean-bib-usage(){ cat << \EOU

tboolean-bib
================

NB : these funcs are parasitic wrt tboolean--



*tboolean-bib* uses the old BoxInBox or PmtInBox 
approach to specifying test geometry. 

This approach it now seldom used : so highly likely to be broken.

However this is still retained as it is expected that the 
old partition-at-intersection approach to geometry is actually 
faster than the new fully general CSG approach. 



PmtInBox

     * see tpmt- for this one

BoxInBox

     * CSG combinations not supported, union/intersection/difference nodes
       appear as placeholder boxes

     * raytrace superficially looks like a union, but on navigating inside 
       its apparent that its just overlapped individual primitives




Historical Note
------------------

CsgInBox

     * DECLARED DEAD, USE PyCsgInBox
     * requires "offsets" identifying node splits into primitives eg offsets=0,1 
     * nodes are specified in tree levelorder, trees must be perfect 
       with 1,3,7 or 15 nodes corresponding to trees of height 0,1,2,3



ISSUE : CURRENTLY ASSERTING FOR LACK OF VOLNAMES
------------------------------------------------------

::

    2017-11-13 15:11:52.471 INFO  [4620770] [*OpticksHub::getGGeoBasePrimary@700] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 0 ggb GGeo
    2017-11-13 15:11:52.472 INFO  [4620770] [*OpticksHub::createTestGeometry@424] OpticksHub::createTestGeometry START
    2017-11-13 15:11:52.475 FATAL [4620770] [GGeoTest::GGeoTest@126] GGeoTest::GGeoTest
    2017-11-13 15:11:52.475 INFO  [4620770] [GGeoTest::init@138] GGeoTest::init START 
    2017-11-13 15:11:52.475 INFO  [4620770] [GPropertyLib::getIndex@345] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Rock]
    2017-11-13 15:11:52.475 ERROR [4620770] [*GMaterialLib::createBufferForTex2d@341] GMaterialLib::createBufferForTex2d NO MATERIALS ?  ni 0 nj 2
    2017-11-13 15:11:52.475 INFO  [4620770] [GPropertyLib::close@396] GPropertyLib::close type GMaterialLib buf NULL
    2017-11-13 15:11:52.475 INFO  [4620770] [GPropertyLib::getIndex@345] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-11-13 15:11:52.475 ERROR [4620770] [*GSurfaceLib::createBufferForTex2d@592] GSurfaceLib::createBufferForTex2d zeros  ni 0 nj 2
    2017-11-13 15:11:52.475 INFO  [4620770] [GPropertyLib::close@396] GPropertyLib::close type GSurfaceLib buf NULL
    2017-11-13 15:11:52.475 INFO  [4620770] [*GGeoTest::makeSolidFromConfig@454] GGeoTest::makeSolidFromConfig i  0 node                  box type  6 csgName             box spec Rock//perfectAbsorbSurface/Vacuum boundary 0 param 0.0000,0.0000,0.0000,1000.0000 trans 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    Assertion failed: (name), function add, file /Users/blyth/opticks/ggeo/GItemList.cc, line 129.
    Process 51948 stopped
    * thread #1: tid = 0x4681e2, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x4681e2, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001020427fa libGGeo.dylib`GItemList::add(this=0x0000000108bc6920, name=0x0000000000000000) + 106 at GItemList.cc:129
        frame #5: 0x00000001021afc59 libGGeo.dylib`GNodeLib::add(this=0x0000000108bc3b10, solid=0x0000000108bc6070) + 1001 at GNodeLib.cc:178
        frame #6: 0x0000000102172b89 libGGeo.dylib`GGeoTest::createBoxInBox(this=0x0000000108bb8b60, solids=0x0000000108bc3ff0) + 505 at GGeoTest.cc:486
        frame #7: 0x00000001021718e7 libGGeo.dylib`GGeoTest::initCreateBIB(this=0x0000000108bb8b60) + 455 at GGeoTest.cc:206
        frame #8: 0x0000000102171278 libGGeo.dylib`GGeoTest::init(this=0x0000000108bb8b60) + 296 at GGeoTest.cc:140
        frame #9: 0x0000000102171139 libGGeo.dylib`GGeoTest::GGeoTest(this=0x0000000108bb8b60, ok=0x0000000105c22400, basis=0x0000000105d13920) + 1865 at GGeoTest.cc:131
        frame #10: 0x00000001021714b5 libGGeo.dylib`GGeoTest::GGeoTest(this=0x0000000108bb8b60, ok=0x0000000105c22400, basis=0x0000000105d13920) + 37 at GGeoTest.cc:132
        frame #11: 0x00000001023085f5 libOpticksGeometry.dylib`OpticksHub::createTestGeometry(this=0x0000000105d0e980, basis=0x0000000105d13920) + 357 at OpticksHub.cc:426
        frame #12: 0x000000010230741c libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x0000000105d0e980) + 844 at OpticksHub.cc:395
        frame #13: 0x0000000102306289 libOpticksGeometry.dylib`OpticksHub::init(this=0x0000000105d0e980) + 137 at OpticksHub.cc:186
        frame #14: 0x0000000102306150 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0e980, ok=0x0000000105c22400) + 464 at OpticksHub.cc:167
        frame #15: 0x00000001023063ad libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x0000000105d0e980, ok=0x0000000105c22400) + 29 at OpticksHub.cc:169
        frame #16: 0x0000000103cac1b6 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe3a8, argc=26, argv=0x00007fff5fbfe488, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #17: 0x0000000103cac61b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe3a8, argc=26, argv=0x00007fff5fbfe488, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #18: 0x000000010000b31d OKTest`main(argc=26, argv=0x00007fff5fbfe488) + 1373 at OKTest.cc:58
        frame #19: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #20: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 6
    frame #6: 0x0000000102172b89 libGGeo.dylib`GGeoTest::createBoxInBox(this=0x0000000108bb8b60, solids=0x0000000108bc3ff0) + 505 at GGeoTest.cc:486
       483          GSolid* solid = makeSolidFromConfig(i);
       484          solids.push_back(solid);  // <-- TODO: eliminate
       485  
    -> 486          m_nodelib->add(solid);
       487      }
       488  }
       489  
    (lldb) p solid
    (GSolid *) $0 = 0x0000000108bc6070
    (lldb) p solid->getPVName()
    (const char *) $1 = 0x0000000000000000
    (lldb) p solid->getLVName()
    (const char *) $2 = 0x0000000000000000
    (lldb) 




EOU
}


tboolean-bib-env(){      olocal- ; tboolean- ;   }
tboolean-bib-dir(){ echo $(opticks-home)/tests ; }
tboolean-bib-cd(){  cd $(tboolean-bib-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }




tboolean-bib-box-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box      parameters=0,0,0,1000               boundary=$(tboolean-container)
                 node=box      parameters=0,0,0,100                boundary=$(tboolean-testobject)

                    )
     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box--(){ echo -n ; }


tboolean-bib-box-small-offset-sphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box-small-offset-sphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-small-offset-sphere-()
{
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=sphere           parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=${1:-difference} parameters=0,0,0,300           boundary=$(tboolean-testobject)
                 node=box              parameters=0,0,0,200           boundary=$(tboolean-testobject)
                 node=sphere           parameters=0,0,200,100         boundary=$(tboolean-testobject)
               )
     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box-small-offset-sphere--(){ echo -n ; }


tboolean-bib-box-sphere-a(){ TESTNAME=${FUNCNAME/-a} tboolean-ana- $* ; } 
tboolean-bib-box-sphere(){ TESTNAME=$FUNCNAME TESTCONFIG=$($FUNCNAME- 2>/dev/null) tboolean-- $* ; }
tboolean-bib-box-sphere-()
{
    local operation=${1:-difference}
    local inscribe=$(python -c "import math ; print 1.3*200/math.sqrt(3)")
    local test_config=(
                 mode=BoxInBox
                 name=$FUNCNAME
                 analytic=1

                 node=box          parameters=0,0,0,1000          boundary=$(tboolean-container)
 
                 node=$operation   parameters=0,0,0,300           boundary=$(tboolean-testobject)
                 node=box          parameters=0,0,0,$inscribe     boundary=$(tboolean-testobject)
                 node=sphere       parameters=0,0,0,200           boundary=$(tboolean-testobject)
               )

     echo "$(join _ ${test_config[@]})" 
}
tboolean-bib-box-sphere--(){ echo -n ; }


