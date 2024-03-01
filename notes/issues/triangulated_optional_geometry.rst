triangulated_optional_geometry
===============================


Whats needed ?
-----------------

high level approach
~~~~~~~~~~~~~~~~~~~~~

* mesh/triangles(vertices+indices) need to be at GAS level 
  and use instancing transforms, avoiding repetition


optional control of mesh usage in geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* need user input control that opts for triangles for particular GAS (or LVID) ? 

  * BUT: the GAS splits are the result of factorization, so in principal
    do not know the indices to pick beforehand 
  * having triangle opted LVID could force solid into separate GAS


creating and persisting mesh triangles (vertices+indices) : DONE: FIRST ATTEMPT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CURRENTLY NOT SUPPRESSING POINTER IN SOLID NAMES FOR NPFOLD KEYS ? 
  POTENTIALLY JUST ISSUE FOR TESTING THAT RUNS FROM GDML NOT LIVE GEOMETRY

* U4Tree/stree needs to use Geant4 polygonization (U4Mesh with U4MESH_EXTRA) 
  to persist the triangles(vertices+indices) (can do this for all solids, as not uploaded)

  * each solid yields an NPFold, eg key name sWorld
    and place to hang the NPFold maybe: "SSim/stree/mesh/sWorld/..." 

  * U4Tree::initSolid looks the place to use U4Mesh::MakeFold


uploading mesh triangles/vertices 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* CSGFoundry model (as it is uploaded) needs to optionally upload only needed triangles
  (vertices+indices), and hold onto device pointers/counts/offsets etc 
  in "CSGMeshSpec" analogous to "CSGPrimSpec"

HMM: SDK/optixMeshViewer uploads just one buffer obtained from GLTF and uses BufferViews 
to access whats needed with appropropriate offsets : makes sense to do similar, but 
manually as only need vertices+indices ?

CSGPrimSpec::

    epsilon:opticks blyth$ opticks-fl CSGPrimSpec
    ./CSGOptiX/GAS_Builder.cc
    ./CSGOptiX/GAS_Builder.h
    ./CSGOptiX/SBT.cc
         SBT::createGAS gets CSGPrimSpec from CSGFoundry 
         and uses GAS_Builder::Build to create GAS

    ./CSG/CSGPrimSpec.cc

    ./CSG/CSGPrim.cc
         CSGPrim::MakeSpec creates CSGPrimSpec

    ./CSG/CSGFoundry.h
         CSGFoundry::getPrimSpecDevice uses above MakeSpec, 
         d_prim with CSGSolid::primOffset CSGSolid::numPrim


HMM: is there need for CSGMesh as well as CSGMeshSpec ? 
Or could add method to CSGSolid/CSGPrim ?

HMM: need to assume all CSGPrim in the CSGSolid 
make the same analytic/triangulated choice, need to enforce that ? 



create OptiX acceleration structures using the mesh triangles/vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

* CSGOptiX needs to use "CSGMeshSpec" to construct triangulated GAS in GAS_Builder


