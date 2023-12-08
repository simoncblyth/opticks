Release_warnings
===================


::

     67%] Linking CXX executable SStrTest
    [ 67%] Linking CXX executable PLogTest
    /home/simon/opticks/sysrap/tests/SPackTest.cc: In function ‘void test_reinterpret_cast()’:
    /home/simon/opticks/sysrap/tests/SPackTest.cc:372:51: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
      372 |     unsigned u2 = reinterpret_cast<unsigned&>(uif.f) ;
          |                                               ~~~~^
    /home/simon/opticks/sysrap/tests/SPackTest.cc: In function ‘void test_reinterpret_cast_arg()’:
    /home/simon/opticks/sysrap/tests/SPackTest.cc:390:56: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
      390 |     dummy_optixTrace(  reinterpret_cast<unsigned&>(uif.f) ) ;
          |                                                    ~~~~^
    [ 67%] Building CXX object tests/CMakeFiles/SGLMTest.dir/SGLMTest.cc.o
    [ 69%] Linking CXX executable SNameVecTest




    [ 19%] Building CXX object CMakeFiles/CSG.dir/CSGDebug_Cylinder.cc.o
    [ 20%] Building CXX object CMakeFiles/CSG.dir/CSG_LOG.cc.o
    In file included from /home/simon/opticks/CSG/CSGQuery.cc:26:
    /home/simon/opticks/CSG/csg_intersect_tree.h: In function ‘float distance_tree(const float3&, const CSGNode*, const float4*, const qat4*)’:
    /home/simon/opticks/CSG/csg_intersect_tree.h:115:12: warning: ‘stack.F4_Stack::data.float4::w’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      115 |     return distance ;
          |            ^~~~~~~~
    /home/simon/opticks/CSG/csg_intersect_tree.h:105:56: warning: ‘stack.F4_Stack::data.float4::z’ may be used uninitialized in this function [-Wmaybe-uninitialized]
      105 |                 case CSG_UNION:        distance = fminf( lhs,  rhs ) ; break ;
          |                                                   ~~~~~^~~~~~~~~~~~~
    [ 21%] Linking CXX shared library libCSG.so
    [ 21%] Built target CSG
    Scanning dependencies of target CSGScanTest
    Scanning dependencies of target CSGFoundry_addPrimNodes_Test

