revive old style tboolean funs
---------------------------------

Overview
----------

Old style tboolean-bib- functions are still a generation ahead 
of the integration tests in geometry config. So as a 
stepping stone to getting integration tests operational 
revive the three remaining old style tboolean funcs.


::

    simon:ggeo blyth$ tboolean-ls | grep bib

    tboolean-bib-box
    tboolean-bib-box-small-offset-sphere
    tboolean-bib-box-sphere



All tboolean styles use ekv config strings
----------------------------------------------

::

    simon:ggeo blyth$ tboolean-testconfig | tr "_" "\n"
    analytic=1
    csgpath=/tmp/blyth/opticks/tboolean-boxsphere--
    name=tboolean-boxsphere--
    mode=PyCsgInBox

    simon:ggeo blyth$ tboolean-box- 2>/dev/null
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-csg-pmt-py_name=tboolean-csg-pmt-py_mode=PyCsgInBox

    
    simon:tests blyth$ tboolean-bib-box-sphere- | tr "_" "\n"   
    mode=BoxInBox
    name=tboolean-bib-box-sphere-
    analytic=1
    node=box
    parameters=0,0,0,1000
    boundary=Rock//perfectAbsorbSurface/Vacuum
    node=difference
    parameters=0,0,0,300
    boundary=Vacuum///GlassSchottF2
    node=box
    parameters=0,0,0,150.111069989
    boundary=Vacuum///GlassSchottF2
    node=sphere
    parameters=0,0,0,200
    boundary=Vacuum///GlassSchottF2



tboolean-bib-box : shows OpenGL viz, raytrace noshow
------------------------------------------------------------

::

    tboolean-bib-box --debugger


    2017-10-19 18:59:33.856 INFO  [406480] [OScene::init@127] OScene::init ggeobase identifier : GGeo
    2017-10-19 18:59:33.856 INFO  [406480] [OGeo::convert@172] OGeo::convert START  numMergedMesh: 1
    2017-10-19 18:59:33.856 INFO  [406480] [GGeoLib::dump@294] OGeo::convert GGeoLib
    2017-10-19 18:59:33.856 INFO  [406480] [GGeoLib::dump@295] GGeoLib TRIANGULATED  numMergedMesh 1 ptr 0x105db6b10
    mm i   0 geocode   A                  numSolids          2 numFaces          24 numITransforms           2 numITransforms*numSolids           4
     num_total_volumes 2 num_instanced_volumes 0 num_global_volumes 2
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::reconstructPartsPerPrim@824] GParts::reconstructPartsPerPrim numParts 2
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::reconstructPartsPerPrim@835] GParts::makePrimBuffer i   0 nodeIndex 4294967295 typ   6 typName box
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::reconstructPartsPerPrim@835] GParts::makePrimBuffer i   1 nodeIndex 4294967295 typ   6 typName box
    2017-10-19 18:59:33.857 WARN  [406480] [GParts::reconstructPartsPerPrim@853] GParts::reconstructPartsPerPrim  non-contiguous node indices nmin 2147483647 nmax 4294967295 num_prim 1 part_per_add.size 2 tran_per_add.size 2
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::fulldump@1278] OGeo::makeAnalyticGeometry --dbganalytic
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::dump@1292] GParts::dump OGeo::makeAnalyticGeometry --dbganalytic
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::dumpPrimInfo@1092] OGeo::makeAnalyticGeometry --dbganalytic (part_offset, parts_for_prim, tran_offset, plan_offset) numPrim:1
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::dumpPrimInfo@1097]  (   0   -2    0    0) 
    2017-10-19 18:59:33.857 INFO  [406480] [GParts::dump@1304] GParts::dump ni 2
         0.0000      0.0000      0.0000   1000.0000 
         0.0000      0.0000     123 <-bnd        0 <-INDEX    bn Rock//perfectAbsorbSurface/Vacuum 
     -1000.0100  -1000.0100  -1000.0100           6 (box) TYPECODE 
      1000.0100   1000.0100   1000.0100          -1 (nodeIndex) 

         0.0000      0.0000      0.0000    100.0000 
         0.0000      0.0000     124 <-bnd        1 <-INDEX    bn Vacuum///GlassSchottF2 
      -100.0100   -100.0100   -100.0100           6 (box) TYPECODE 
       100.0100    100.0100    100.0100          -1 (nodeIndex) 

    2017-10-19 18:59:33.857 INFO  [406480] [GParts::Summary@1104] OGeo::makeAnalyticGeometry --dbganalytic num_parts 2 num_prim 1
    4294967295 :  2 
     part  0 : node 4294967295 type  6 boundary [123] Rock//perfectAbsorbSurface/Vacuum  
     part  1 : node 4294967295 type  6 boundary [124] Vacuum///GlassSchottF2  
    2017-10-19 18:59:33.857 INFO  [406480] [NPY<float>::dump@1350] partBuf (2,4,4) 

    (  0)       0.000       0.000       0.000    1000.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  0)   -1000.010   -1000.010   -1000.010       0.000 
    (  0)    1000.010    1000.010    1000.010         nan 
    (  1)       0.000       0.000       0.000     100.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)    -100.010    -100.010    -100.010       0.000 
    (  1)     100.010     100.010     100.010         nan 
    2017-10-19 18:59:33.857 INFO  [406480] [NPY<int>::dump@1350] primBuf:partOffset/numParts/primIndex/0 (1,4) 

    (  0)           0          -2           0           0 
    2017-10-19 18:59:33.857 FATAL [406480] [OGeo::makeAnalyticGeometry@662]  MISMATCH (numPrim != numSolids)  numSolids 2 numPrim 1 numPart 2 numTran 0 numPlan 0
    Assertion failed: (match && "Sanity check failed "), function makeAnalyticGeometry, file /Users/blyth/opticks/optixrap/OGeo.cc, line 670.
    Process 4754 stopped
    * thread #1: tid = 0x633d0, 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8b576866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8b576866:  jae    0x7fff8b576870            ; __pthread_kill + 20
       0x7fff8b576868:  movq   %rax, %rdi
       0x7fff8b57686b:  jmp    0x7fff8b573175            ; cerror_nocancel
       0x7fff8b576870:  retq   
    (lldb) 




Small glass box inside large vacuum box with perfectAbsorbSurface::

    simon:opticks blyth$ tboolean-;tboolean-bib-box- | tr "_" "\n"
    mode=BoxInBox
    name=tboolean-bib-box-
    analytic=1

    node=box
    parameters=0,0,0,1000
    boundary=Rock//perfectAbsorbSurface/Vacuum

    node=box
    parameters=0,0,0,100
    boundary=Vacuum///GlassSchottF2





CSG_PARTLIST
-------------


::

    simon:sysrap blyth$ opticks-find CSG_PARTLIST
    ./cfg4/CMaker.cc:        case CSG_PARTLIST:
    ./ggeo/GGeoTest.cc:        if(flags == CSG_PARTLIST) primIdx++ ;   // constituents dont merit new primIdx
    ./ggeo/GMaker.cc:         case CSG_PARTLIST:
    ./ggeo/GSolid.cc:         m_csgflag(CSG_PARTLIST),
    ./sysrap/OpticksCSG.h:    CSG_PARTLIST=4,   
    ./sysrap/OpticksCSG.h:static const char* CSG_PARTLIST_      = "partlist" ; 
    ./sysrap/OpticksCSG.h:    else if(strcmp(nodename, CSG_PARTLIST_) == 0)       tc = CSG_PARTLIST ;
    ./sysrap/OpticksCSG.h:        case CSG_PARTLIST:      s = CSG_PARTLIST_      ; break ; 
    ./bin/c_enums_to_python.py:        static const char* CSG_PARTLIST_      = "partlist" ; 
    simon:opticks blyth$ 



::

     03 typedef enum {
      4     CSG_ZERO=0,
      5     CSG_UNION=1,
      6     CSG_INTERSECTION=2,
      7     CSG_DIFFERENCE=3,
      8     CSG_PARTLIST=4,
      9 
     10     CSG_SPHERE=5,
     11        CSG_BOX=6,
     12    CSG_ZSPHERE=7,
     13      CSG_ZLENS=8,
     14        CSG_PMT=9,
     15      CSG_PRISM=10,
     16       CSG_TUBS=11,
     17   CSG_CYLINDER=12,
     18       CSG_SLAB=13,
     19      CSG_PLANE=14,
     20 
     21       CSG_CONE=15,
     22  CSG_MULTICONE=16,
     23       CSG_BOX3=17,
     24  CSG_TRAPEZOID=18,
     25  CSG_CONVEXPOLYHEDRON=19,
     26      CSG_DISC=20,
     27    CSG_SEGMENT=21,
     28    CSG_ELLIPSOID=22,
     29    CSG_TORUS=23,
     30    CSG_HYPERBOLOID=24,
     31    CSG_CUBIC=25,
     32  CSG_UNDEFINED=26,
     33 
     34  CSG_FLAGPARTLIST=100,
     35  CSG_FLAGNODETREE=101,
     36  CSG_FLAGINVISIBLE=102
     37 
     38 } OpticksCSG_t ;
     39 




