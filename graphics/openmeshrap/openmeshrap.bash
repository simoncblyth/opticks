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
openmeshrap-idir(){ echo $(opticks-idir) ; }
openmeshrap-bdir(){ echo $(opticks-bdir)/$(openmeshrap-rel) ; }

openmeshrap-cd(){   cd $(openmeshrap-dir); }
openmeshrap-scd(){  cd $(openmeshrap-sdir); }
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





