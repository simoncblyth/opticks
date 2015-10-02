# === func-gen- : graphics/openmeshrap/openmeshrap fgp graphics/openmeshrap/openmeshrap.bash fgn openmeshrap fgh graphics/openmeshrap
openmeshrap-src(){      echo graphics/openmeshrap/openmeshrap.bash ; }
openmeshrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openmeshrap-src)} ; }
openmeshrap-vi(){       vi $(openmeshrap-source) ; }
openmeshrap-env(){      elocal- ; }
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



EOU
}
openmeshrap-dir(){  echo $(env-home)/graphics/openmeshrap ; }
openmeshrap-idir(){ echo $(local-base)/env/graphics/openmeshrap ; }
openmeshrap-bdir(){ echo $(openmeshrap-idir).build ; }

openmeshrap-cd(){   cd $(openmeshrap-dir); }
openmeshrap-icd(){  cd $(openmeshrap-idir); }
openmeshrap-bcd(){  cd $(openmeshrap-bdir); }
openmeshrap-bin(){  echo $(openmeshrap-idir)/bin/OpenMeshRapTest ; }

openmeshrap-wipe(){
  local bdir=$(openmeshrap-bdir)
  rm -rf $bdir 

}

openmeshrap-cmake(){
  local iwd=$PWD
  local bdir=$(openmeshrap-bdir)
  mkdir -p $bdir
  openmeshrap-bcd

  cmake $(openmeshrap-dir) \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=$(openmeshrap-idir) 

  cd $iwd
}

openmeshrap-make(){
  local iwd=$PWD
  openmeshrap-bcd
  make $*
  cd $iwd
}

openmeshrap-install(){
  openmeshrap-make install
}

openmeshrap--(){
  openmeshrap-cmake
  openmeshrap-make
  openmeshrap-install
}

