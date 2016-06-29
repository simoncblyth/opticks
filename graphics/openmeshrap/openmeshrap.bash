# === func-gen- : graphics/openmeshrap/openmeshrap fgp graphics/openmeshrap/openmeshrap.bash fgn openmeshrap fgh graphics/openmeshrap
openmeshrap-rel(){      echo graphics/openmeshrap ; }
openmeshrap-src(){      echo graphics/openmeshrap/openmeshrap.bash ; }
openmeshrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openmeshrap-src)} ; }
openmeshrap-vi(){       vi $(openmeshrap-source) ; }
openmeshrap-usage(){ cat << EOU

OpenMeshRap
=============

Developing mesh surgery code.


Where to do the surgery in ggv ?
----------------------------------

Add AssimpGGeo methods to be invoked prior to convertMeshes that operate at ai level::

    AssimpGGeo::checkMeshes 
    AssimpGGeo::fixMeshes  
    AssimpGGeo::convertMeshes 



Fix is failing
---------------

::

    op --dbg -G --meshrap trace --ggeo trace --asirap trace


    2016-06-29 18:41:21.562 INFO  [13234830] [*MTool::joinSplitUnion@82] MTool::joinSplitUnion  index 24 shortname iav
    2016-06-29 18:41:21.562 VERB  [13234830] [>::load@49] MWrap<MeshT>::load NumVertices 148 NumFaces 288
    2016-06-29 18:41:21.562 VERB  [13234830] [>::copyIn@63] MWrap<MeshT>::load num_vertices 148 num_faces 288
    2016-06-29 18:41:21.565 DEBUG [13234830] [>::labelSpatialPairs@324]   0 (  0,168) an                0.000 -0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                3.376 -0.503 0.050
    2016-06-29 18:41:21.565 DEBUG [13234830] [>::labelSpatialPairs@324]   1 (  1,169) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                3.376 0.359 0.050
    2016-06-29 18:41:21.565 DEBUG [13234830] [>::labelSpatialPairs@324]   2 ( 10,170) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                3.152 1.193 0.050
    2016-06-29 18:41:21.566 DEBUG [13234830] [>::labelSpatialPairs@324]   3 ( 14,171) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                2.719 1.939 0.050
    2016-06-29 18:41:21.566 DEBUG [13234830] [>::labelSpatialPairs@324]   4 ( 18,172) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                2.108 2.549 0.050
    2016-06-29 18:41:21.566 DEBUG [13234830] [>::labelSpatialPairs@324]   5 ( 22,173) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                1.362 2.983 0.050
    2016-06-29 18:41:21.566 DEBUG [13234830] [>::labelSpatialPairs@324]   6 ( 26,174) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                0.528 3.206 0.050
    2016-06-29 18:41:21.567 DEBUG [13234830] [>::labelSpatialPairs@324]   7 ( 30,175) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -0.335 3.206 0.050
    2016-06-29 18:41:21.567 DEBUG [13234830] [>::labelSpatialPairs@324]   8 ( 34,176) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -1.168 2.983 0.050
    2016-06-29 18:41:21.567 DEBUG [13234830] [>::labelSpatialPairs@324]   9 ( 38,177) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -1.915 2.549 0.050
    2016-06-29 18:41:21.567 DEBUG [13234830] [>::labelSpatialPairs@324]  10 ( 42,178) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -2.524 1.939 0.050
    2016-06-29 18:41:21.567 DEBUG [13234830] [>::labelSpatialPairs@324]  11 ( 46,179) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -2.958 1.193 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  12 ( 50,180) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -3.181 0.360 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  13 ( 54,181) an                0.000 0.000 1.000 bn               -0.000 0.000 -1.000 a.b     -1.000 dp               -3.181 -0.503 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  14 ( 58,182) an               -0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -2.958 -1.337 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  15 ( 62,183) an               -0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -2.524 -2.081 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  16 ( 66,184) an               -0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -1.915 -2.691 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  17 ( 70,185) an               -0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp               -1.168 -3.127 0.050
    2016-06-29 18:41:21.568 DEBUG [13234830] [>::labelSpatialPairs@324]  18 ( 74,186) an               -0.000 0.000 1.000 bn                0.000 -0.000 -1.000 a.b     -1.000 dp               -0.335 -3.351 0.050
    2016-06-29 18:41:21.569 DEBUG [13234830] [>::labelSpatialPairs@324]  19 ( 78,187) an                0.000 0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                0.528 -3.351 0.050
    2016-06-29 18:41:21.569 DEBUG [13234830] [>::labelSpatialPairs@324]  20 ( 82,188) an                0.000 -0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                1.362 -3.127 0.050
    2016-06-29 18:41:21.569 DEBUG [13234830] [>::labelSpatialPairs@324]  21 ( 86,189) an                0.000 -0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                2.108 -2.691 0.050
    2016-06-29 18:41:21.569 DEBUG [13234830] [>::labelSpatialPairs@324]  22 ( 90,190) an                0.000 -0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                2.719 -2.081 0.050
    2016-06-29 18:41:21.569 DEBUG [13234830] [>::labelSpatialPairs@324]  23 ( 94,191) an                0.000 -0.000 1.000 bn                0.000 0.000 -1.000 a.b     -1.000 dp                3.152 -1.337 0.050
    2016-06-29 18:41:21.569 INFO  [13234830] [>::labelSpatialPairs@345] MWrap<MeshT>::labelSpatialPairs fposprop centroid fpropname paired npair 24
    Assertion failed: (p != NULL), function property, file /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/../../OpenMesh/Core/Utils/PropertyContainer.hh, line 158.
    Process 13243 stopped
    * thread #1: tid = 0xc9f28e, 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8ea02866:  jae    0x7fff8ea02870            ; __pthread_kill + 20
       0x7fff8ea02868:  movq   %rax, %rdi
       0x7fff8ea0286b:  jmp    0x7fff8e9ff175            ; cerror_nocancel
       0x7fff8ea02870:  retq   
    (lldb) 





::

    op -G --dbg

    016-06-29 16:30:30.528 INFO  [13191086] [AssimpGGeo::convertSensors@569] AssimpGGeo::convertSensors gss GSS:: GPropertyMap<T>:: 79    skinsurface s: GOpticalSurface  type 0 model 1 finish 3 value     1 lvPmtHemiCathodeSensorSurface k:ABSLENGTH EFFICIENCY RINDEX
    2016-06-29 16:30:30.528 INFO  [13191086] [AssimpGGeo::convertMeshes@643] AssimpGGeo::convertMeshes NumMeshes 249
    2016-06-29 16:30:30.538 INFO  [13191086] [*MTool::joinSplitUnion@82] MTool::joinSplitUnion  index 24 shortname iav
    2016-06-29 16:30:30.549 INFO  [13191086] [>::labelSpatialPairs@331] MWrap<MeshT>::labelSpatialPairs fposprop centroid fpropname paired npair 24
    Assertion failed: (p != NULL), function property, file /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/../../OpenMesh/Core/Utils/PropertyContainer.hh, line 158.
    /Users/blyth/env/bin/op.sh: line 374:  2336 Abort trap: 6           /usr/local/opticks/lib/GGeoViewTest -G


    (lldb) bt
    * thread #1: tid = 0xc94eeb, 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8609f35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8cdefb1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8cdb99bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000101dba50d libOpenMeshCored.4.1.dylib`OpenMesh::PropertyT<OpenMesh::Attributes::StatusInfo>& OpenMesh::PropertyContainer::property<OpenMesh::Attributes::StatusInfo>(this=0x00000001295e54f8, _h=BasePropHandleT<OpenMesh::Attributes::StatusInfo> at 0x00007fff5fbfb750) + 525 at PropertyContainer.hh:158
        frame #5: 0x0000000101dba2db libOpenMeshCored.4.1.dylib`OpenMesh::FPropHandleT<OpenMesh::Attributes::StatusInfo>::reference OpenMesh::BaseKernel::property<OpenMesh::Attributes::StatusInfo>(this=0x00000001295e5490, _ph=FPropHandleT<OpenMesh::Attributes::StatusInfo> at 0x00007fff5fbfb7b8, _fh=FaceHandle at 0x00007fff5fbfb7b0) + 43 at BaseKernel.hh:390
        frame #6: 0x0000000101db8be3 libOpenMeshCored.4.1.dylib`OpenMesh::ArrayKernel::status(this=0x00000001295e5490, _fh=OpenMesh::ArrayKernel::FaceHandle at 0x00007fff5fbfb7e8) + 51 at ArrayKernel.hh:485
        frame #7: 0x0000000101dccfa6 libOpenMeshCored.4.1.dylib`OpenMesh::PolyConnectivity::delete_face(this=0x00000001295e5490, _fh=OpenMesh::ArrayKernel::FaceHandle at 0x00007fff5fbfbaf8, _delete_isolated_vertices=true) + 134 at PolyConnectivity.cc:513
        frame #8: 0x0000000101f1174a libOpenMeshRap.dylib`MWrap<OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DefaultTraits> >::deleteFaces(this=0x00007fff5fbfc468, fpredicate_name=0x0000000101f5969d) + 602 at MWrap.cc:359
        frame #9: 0x0000000101f3c8b6 libOpenMeshRap.dylib`MTool::joinSplitUnion(gmesh=0x00000001295e51d0, opticks=0x0000000106220580) + 1606 at MTool.cc:130
        frame #10: 0x00000001020f8b8b libGGeo.dylib`GGeo::invokeMeshJoin(this=0x000000010644a550, mesh=0x00000001295e51d0) + 91 at GGeo.cc:1741
        frame #11: 0x0000000101cae132 libAssimpRap.dylib`AssimpGGeo::convertMeshes(this=0x00007fff5fbfd078, scene=0x000000010fbb1120, gg=0x000000010644a550, (null)=0x00007fff5fbfe943) + 2978 at AssimpGGeo.cc:703
        frame #12: 0x0000000101cab84a libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007fff5fbfd078, ctrl=0x00007fff5fbfe943) + 362 at AssimpGGeo.cc:167
        frame #13: 0x0000000101cab664 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x000000010644a550) + 1156 at AssimpGGeo.cc:153
        frame #14: 0x00000001020f03fb libGGeo.dylib`GGeo::loadFromG4DAE(this=0x000000010644a550) + 251 at GGeo.cc:556
        frame #15: 0x00000001020f01c5 libGGeo.dylib`GGeo::loadGeometry(this=0x000000010644a550) + 341 at GGeo.cc:536
        frame #16: 0x000000010220a1f4 libOpticksGeometry.dylib`OpticksGeometry::loadGeometryBase(this=0x0000000106429cd0) + 1172 at OpticksGeometry.cc:190
        frame #17: 0x0000000102209ad6 libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000106429cd0) + 294 at OpticksGeometry.cc:139
        frame #18: 0x0000000104444dcb libGGeoView.dylib`App::loadGeometry(this=0x00007fff5fbfe2b8) + 123 at App.cc:417
        frame #19: 0x000000010000bdaa GGeoViewTest`main(argc=3, argv=0x00007fff5fbfe440) + 1306 at GGeoViewTest.cc:69
        frame #20: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 13
    frame #13: 0x0000000101cab664 libAssimpRap.dylib`AssimpGGeo::load(ggeo=0x000000010644a550) + 1156 at AssimpGGeo.cc:153
       150  
       151      agg.setVerbosity(verbosity);
       152  
    -> 153      int rc = agg.convert(ctrl);
       154  
       155      return rc ;
       156  }

    (lldb) f 12
    frame #12: 0x0000000101cab84a libAssimpRap.dylib`AssimpGGeo::convert(this=0x00007fff5fbfd078, ctrl=0x00007fff5fbfe943) + 362 at AssimpGGeo.cc:167
       164  
       165      convertMaterials(scene, m_ggeo, ctrl );
       166      convertSensors( m_ggeo ); 
    -> 167      convertMeshes(scene, m_ggeo, ctrl);
       168      convertStructure(m_ggeo);

    (lldb) f 11
    frame #11: 0x0000000101cae132 libAssimpRap.dylib`AssimpGGeo::convertMeshes(this=0x00007fff5fbfd078, scene=0x000000010fbb1120, gg=0x000000010644a550, (null)=0x00007fff5fbfe943) + 2978 at AssimpGGeo.cc:703
       700  
       701          gmesh->setName(meshname);
       702  
    -> 703          GMesh* gfixed = gg->invokeMeshJoin(gmesh);  
       704   
       705          assert(gfixed) ; 

    (lldb) f 10
    frame #10: 0x00000001020f8b8b libGGeo.dylib`GGeo::invokeMeshJoin(this=0x000000010644a550, mesh=0x00000001295e51d0) + 91 at GGeo.cc:1741
       1738     bool join = shouldMeshJoin(mesh);
       1739     if(join)
       1740     {
    -> 1741         result = (*m_join_imp)(mesh, m_opticks); 
       1742 
       1743         result->setName(mesh->getName()); 
       1744         result->setIndex(mesh->getIndex()); 

    (lldb) f 9
    frame #9: 0x0000000101f3c8b6 libOpenMeshRap.dylib`MTool::joinSplitUnion(gmesh=0x00000001295e51d0, opticks=0x0000000106220580) + 1606 at MTool.cc:130
       127  
       128      MWrap<MyMesh>::labelSpatialPairs( wa.getMesh(), wb.getMesh(), delta, "centroid", "paired");
       129  
    -> 130      wa.deleteFaces("paired");
       131      wb.deleteFaces("paired");
       132  
       133      wa.collectBoundaryLoop();

    (lldb) f 8
    frame #8: 0x0000000101f1174a libOpenMeshRap.dylib`MWrap<OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DefaultTraits> >::deleteFaces(this=0x00007fff5fbfc468, fpredicate_name=0x0000000101f5969d) + 602 at MWrap.cc:359
       356          {
       357             // std::cout << f->idx() << " " ; 
       358              bool delete_isolated_vertices = true ; 
    -> 359              mesh->delete_face( *f, delete_isolated_vertices );
       360              count++ ; 
       361          }

    (lldb) f 8
    frame #8: 0x0000000101f1174a libOpenMeshRap.dylib`MWrap<OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DefaultTraits> >::deleteFaces(this=0x00007fff5fbfc468, fpredicate_name=0x0000000101f5969d) + 602 at MWrap.cc:359
       356          {
       357             // std::cout << f->idx() << " " ; 
       358              bool delete_isolated_vertices = true ; 
    -> 359              mesh->delete_face( *f, delete_isolated_vertices );
       360              count++ ; 
       361          }
       362      }
    (lldb) f 7
    frame #7: 0x0000000101dccfa6 libOpenMeshCored.4.1.dylib`OpenMesh::PolyConnectivity::delete_face(this=0x00000001295e5490, _fh=OpenMesh::ArrayKernel::FaceHandle at 0x00007fff5fbfbaf8, _delete_isolated_vertices=true) + 134 at PolyConnectivity.cc:513
       510  
       511  void PolyConnectivity::delete_face(FaceHandle _fh, bool _delete_isolated_vertices)
       512  {
    -> 513    assert(_fh.is_valid() && !status(_fh).deleted());
       514  
       515    // mark face deleted
       516    status(_fh).set_deleted(true);

    (lldb) expr _fh.is_valid()
    (bool) $2 = true

    (lldb) expr status(_fh)
    Assertion failed: (p != NULL), function property, file /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/../../OpenMesh/Core/Utils/PropertyContainer.hh, line 158.
    error: Execution was interrupted, reason: signal SIGABRT.
    The process has been returned to the state before expression evaluation.


Test fails in same place::

    simon:openmeshrap blyth$ lldb OpenMeshRapTest -- --meshrap trace
    (lldb) target create "OpenMeshRapTest"
    Current executable set to 'OpenMeshRapTest' (x86_64).
    (lldb) settings set -- target.run-args  "--meshrap" "trace"
    (lldb) r
    Process 4726 launched: '/usr/local/opticks/lib/OpenMeshRapTest' (x86_64)
    2016-06-29 16:58:27.078 INFO  [13201257] [main@37] OpenMeshRapTest
    2016-06-29 16:58:27.079 INFO  [13201257] [main@42] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2016-06-29 16:58:28.616 INFO  [13201257] [main@46]  after load_deduped 
    2016-06-29 16:58:28.616 INFO  [13201257] [GMesh::Summary@1168] GMesh::Summary
    GMesh::Summary idx 0 vx 212995 fc 434816 n (null) sn (null) 
    center_extent -16520.000 -802110.000  -7125.000   7710.562 
    GMesh::Summary
     a   7710.562      0.000      0.000 -16520.000 
     b      0.000   7710.562      0.000 -802110.000 
     c      0.000      0.000   7710.562  -7125.000 
     d      0.000      0.000      0.000      1.000 
    2016-06-29 16:58:28.616 VERB  [13201257] [>::load@49] MWrap<MeshT>::load NumVertices 212995 NumFaces 434816
    2016-06-29 16:58:28.617 VERB  [13201257] [>::copyIn@63] MWrap<MeshT>::load num_vertices 212995 num_faces 434816
    Assertion failed: (p != NULL), function property, file /usr/local/opticks/externals/include/OpenMesh/Core/Utils/PropertyContainer.hh, line 158.
    Process 4726 stopped
    * thread #1: tid = 0xc96f69, 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8ea02866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8ea02866:  jae    0x7fff8ea02870            ; __pthread_kill + 20
       0x7fff8ea02868:  movq   %rax, %rdi
       0x7fff8ea0286b:  jmp    0x7fff8e9ff175            ; cerror_nocancel
       0x7fff8ea02870:  retq   


Note far too many Vertices and Faces. Mesh fixing was intended for small sub-meshes not entire merged meshes.
So maybe issue is a mis-use one.



Test is failing
------------------

::

    simon:env blyth$ op --openmesh

    ...
    PolyMeshT::add_face: complex edge
    PolyMeshT::add_face: complex edge
    Assertion failed: (boundary_prev.is_valid()), function add_face, file /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/PolyConnectivity.cc, line 276.
    /Users/blyth/env/bin/op.sh: line 374: 82728 Abort trap: 6           /usr/local/opticks/bin/OpenMeshRapTest --openmesh
    simon:env blyth$ 


    simon:env blyth$ opticks-ctest -R OpenMeshRapTest.OpenMeshRapTest -V

    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex vertex
    42: PolyMeshT::add_face: complex edge
    42: PolyMeshT::add_face: complex edge
    42: PolyMeshT::add_face: complex edge
    42: PolyMeshT::add_face: complex edge
    42: Assertion failed: (boundary_prev.is_valid()), function add_face, file /usr/local/opticks/externals/openmesh/OpenMesh-4.1/src/OpenMesh/Core/Mesh/PolyConnectivity.cc, line 276.
    1/1 Test #42: OpenMeshRapTest.OpenMeshRapTest ...***Exception: Other  1.95 sec

    0% tests passed, 1 tests failed out of 1

    Total Test time (real) =   1.95 sec

    The following tests FAILED:
         42 - OpenMeshRapTest.OpenMeshRapTest (OTHER_FAULT)
    Errors while running CTest





TODO
-----

#. convert surgeried back into NPY format 

#. rearrange code into lib for actual usage

#. test in ggv-


Done
-----

#. convert NPY meshes into OpenMesh 
#. partition G4 created triangle soup (did Assimp do any diddling?) 
   into real connected V - E + F = 2  Euler Polyhedra 

#. geometrical comparison of two component meshes to identify close faces, 
   
   * for each face of mesh A find parallel faces using 
     absolute normal dot products in mesh B
   * compare distances in normal direction between candidate aligned faces 
   * also check standard distance between barycenters of candidates
   * devise criteria to pick the cleaved faces of mesh A and B

#. delete the cleaved faces from A and B, make sure mesh boundary is 
   navigable

#. find way to weld together A and B by joining up the boundary
   need distance criterior to decide whether to fuse or put in 
   edges to do the welding  

   * de-nudge in z ?

   * suspect the volume in question will need new edges going
     outwards in (x,y) almost fixed in z : as 
     the cleave happened along a z flange plane 
     (polycone with 2 identical z planes)



Following Mesh Boundary
------------------------

* https://mailman.rwth-aachen.de/pipermail/openmesh/2009-April/000235.html

::

    your code will give you the ccw order only at interior.
    but you have to check for the bounday, because at bounday, the case is like:
         h0                     h1                         h2
    ------------------\ vh0 -------------------\ vh1 --------------\ vh2

    TriMesh::HHandle heh = patchmesh->halfedge_handle(vh);
    if (patchmesh->is_boundary(heh)) {
    heh = patchmesh->opposite_halfedge_handle((patchmesh->prev_halfedge_handle(heh));
    vh = patchmesh->to_vertex_handle(heh);
    ...
    }
    else {
    heh = patchmesh->next_halfedge_handle(heh);
    vh = patchmesh->to_vertex_handle(heh);
    ....
    }


* https://mailman.rwth-aachen.de/pipermail/openmesh/2007-November/000051.html


Bill,
   Thank you for your replyment so quick! yes, I have just look the
adjust_outgoing_halfedge function, it sets the vertexhandle point to
the outgoing halfedge:-)

Ps. The following code may be interesting.

::

      std::vector<Mesh::VertexHandle>  loop;

      // find 1st boundary vertex
      for (v_it=mesh_.vertices_begin(); v_it!=v_end; ++v_it)
        if (mesh_.is_boundary(v_it.handle()))
          break;

      // boundary found ?
      if (v_it == v_end)
      {
        std::cerr << "No boundary found\n";
        return;
      }

      // collect boundary loop
      vh = v_it.handle();
      hh = mesh_.halfedge_handle(vh);
      do
      {
        loop.push_back(mesh_.to_vertex_handle(hh));
        hh = mesh_.next_halfedge_handle(hh);
      }
      while (hh != mesh_.halfedge_handle(vh));




Combining Meshes
------------------

* https://mailman.rwth-aachen.de/pipermail/openmesh/2010-March/000405.html

The easiest way to do it is to create a map while adding the vertices to the 
other mesh, mapping from the old mesh vertex handle to the new mesh vertex 
handle. Than you can just iterate over all faces of the old mesh, use a 
FaceVertex iterator and add the face with the mapped vertex handles to the new 
mesh.


Windows Libs
--------------

Bizarre, test runs but no libOpenMesh ?

::

    $ ldd $(which OpenMeshRapTest.exe) | grep opticks
            libOpenMeshRap.dll => /usr/local/opticks/lib/libOpenMeshRap.dll (0x6dac0000)
            libGGeo.dll => /usr/local/opticks/lib/libGGeo.dll (0x69740000)
            libOpticksCore.dll => /usr/local/opticks/lib/libOpticksCore.dll (0x623c0000)
            libBCfg.dll => /usr/local/opticks/lib/libBCfg.dll (0x65180000)
            libBRegex.dll => /usr/local/opticks/lib/libBRegex.dll (0x6cbc0000)
            libNPY.dll => /usr/local/opticks/lib/libNPY.dll (0x20b0000)






EOU
}
openmeshrap-env(){      elocal- ; opticks- ;  }
openmeshrap-dir(){  echo $(env-home)/graphics/openmeshrap ; }
openmeshrap-sdir(){ echo $(env-home)/graphics/openmeshrap ; }
openmeshrap-tdir(){ echo $(env-home)/graphics/openmeshrap/tests ; }
openmeshrap-idir(){ echo $(opticks-idir) ; }
openmeshrap-bdir(){ echo $(opticks-bdir)/$(openmeshrap-rel) ; }

openmeshrap-cd(){   cd $(openmeshrap-dir); }
openmeshrap-scd(){  cd $(openmeshrap-sdir); }
openmeshrap-tcd(){  cd $(openmeshrap-tdir); }
openmeshrap-icd(){  cd $(openmeshrap-idir); }
openmeshrap-bcd(){  cd $(openmeshrap-bdir); }

openmeshrap-bin(){  echo $(openmeshrap-idir)/bin/OpenMeshRapTest ; }

openmeshrap-wipe(){
  local bdir=$(openmeshrap-bdir)
  rm -rf $bdir 

}


openmeshrap-name(){ echo OpenMeshRap ; }
openmeshrap-tag(){  echo MESHRAP ; }


openmeshrap--(){        opticks--     $(openmeshrap-bdir) ; }
openmeshrap-ctest(){    opticks-ctest $(openmeshrap-bdir) $* ; }
openmeshrap-genproj(){  openmeshrap-scd ; opticks-genproj $(openmeshrap-name) $(openmeshrap-tag) ; }
openmeshrap-gentest(){  openmeshrap-tcd ; opticks-gentest ${1:-AssimpGGeo} $(openmeshrap-tag) ; }

openmeshrap-txt(){ vi $(openmeshrap-sdir)/CMakeLists.txt  $(openmeshrap-tdir)/CMakeLists.txt ; }



