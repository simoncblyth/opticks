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





    [ 18%] Building CXX object CMakeFiles/CSG.dir/CSGScan.cc.o
    [ 19%] Building CXX object CMakeFiles/CSG.dir/CSGRecord.cc.o
    [ 20%] Building CXX object CMakeFiles/CSG.dir/CSGSimtraceRerun.cc.o
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
    Scanning dependencies of target intersect_leaf_cylinder_test
    Scanning dependencies of target CSGSolidTest
    Scanning dependencies of target CSGPrimTest







    [ 19%] Building CXX object CMakeFiles/U4.dir/ShimG4OpAbsorption.cc.o
    [ 20%] Building CXX object CMakeFiles/U4.dir/U4Physics.cc.o
    /home/simon/opticks/u4/U4.cc: In function ‘quad6 MakeGenstep_DsG4Scintillation_r4695(const G4Track*, const G4Step*, G4int, G4int, G4double)’:
    /home/simon/opticks/u4/U4.cc:105:27: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
      105 |     sscint& gs = (sscint&)_gs ;
          |                           ^~~
    /home/simon/opticks/u4/U4.cc: In function ‘quad6 MakeGenstep_G4Cerenkov_modified(const G4Track*, const G4Step*, G4int, G4double, G4double, G4double, G4double, G4double, G4double, G4double)’:
    /home/simon/opticks/u4/U4.cc:215:33: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
      215 |     scerenkov& gs = (scerenkov&)_gs ;
          |                                 ^~~


