Review Progress and Where Next ?
===================================


Immediate
-----------

* go thru tboolean checking/investigating any test geometry mismatches
* flesh out tgltf- with some intermediate geometry trees specified
  in python




Recent Progress
-----------------

Opticks now supports:

* ray tracing of CSG node trees using GPU implementation of evaluative CSG 
  that applies directly to serialized node trees copied to GPU   

* CPU polygonization of CSG node trees using signed distance functions,
  providing fast OpenGL rasterized visualization 

* Generalized transforms (Translate-Rotate-Scale) of 
  any node of the CSG tree, so shapes such as ellipsoids 
  can be derived by non-uniform scaling without 
  implementing separate primitives to support ellipsoids. 

* Small number of primitives: sphere, box, truncated sphere, cylinder, truncated cone, convex polyhedron

* Interactive composited visualization of the ray traced images and rasterized 
  polygonization meshes together with photon propagations thru the geometry 
  for convenient geometry debugging.


TODO : j1707 review size of GPU buffers, any more instancing possible ? 
--------------------------------------------------------------------------

* potential reorg optix geometry into more objects 


TODO : j1707 deep trees, expand balancing to handle ?
---------------------------------------------------------


TODO: get verbosity under control in GDML/glTF route especially GParts 
-------------------------------------------------------------------------


TODO: deep csg trees are still being skipped
--------------------------------------------------

Most of these are differences...

* complement was implemented to allow tree rearrangement, use that 
  to create +ve form tree (with signs all in the leaves as complements) 
  so then have a fully commutative CSG expression tree 
  that can easily be balanced


TODO: polycone testing, general coincident surface testing(how?)
------------------------------------------------------------------

Polycone is implemented as "uniontree" ... suspect
may need some z-join-nudging to avoid incorrect internal
surfaces arising due to coincident surfaces.


TODO: zero-repetition scene data structure 
--------------------------------------------


TODO: analytic path caching
-----------------------------


TODO: thin solid polygonization
----------------------------------

* eg cathode


TODO : Push CSG node tree support thru to cfg4
------------------------------------------------

* :doc:`ncsg_support_in_cfg4`

Add:

* creation of Geant4 geometries from the CSG node tree description.
* comparisons of GPU and CPU propagations using CSG node tree geometries




NOPE : More Primitives
------------------------

Best to let the actual needs of detector geometries 
to drive the implementation of primitives, otherwise
becomes a never ending task... 


What is needed to add primitives
----------------------------------

* sysrap/OpticksCSG_t type code and name

* Within csg_intersect_part.h:

  * CUDA/OptiX bounds calculation
  * CUDA/OptiX intersection calculation 

* npy- nnode subclass for the primitive including 
  signed distance function for the primitive shape
  (can use min, max) to do sneaky CSG inside the 
  primitive eg for finite cylinder  




NOPE: Add NPolygonization of partlist ?
--------------------------------------------

World allow cleaning up the currently dirty GMaker/PmtInBox mode, 
which makes the adhoc association of a loaded PMT mesh 
with analytic part list.  

YES BUT, partlist are very limited, only keep them around as 
a possible optimization of csg tree, so this is too much of a cul-de-sac.



What is NOT needed for each primitive ?
-------------------------------------------

* Do not need a parametrized surface description to generate a mesh,
  this is due to the use of generalized polygonization isosurface extraction 
  using signed distance functions which can be CSG combined with min/max. 





DONE: tboolean-pmt converts ddbase.py PMT into NCSG node tree 
---------------------------------------------------------------

See pmt-ecd

Currently:

    pmt- parses detdesc PMT description (using python/lxml) 
    into a python node tree and creates the CSG_FLAGPARTLIST 
    serialization (the z-partitioned solids) 
    tpmt- uses in the PmtInBox mode

Instead:

     Skip the non-generalizable partitioning by adding support 
     for converting the python node tree into CSG_FLAGNODETREE  
     serialization.


This should allow rapid testing of a CSG node tree 
description of the PMT geometry. 

Also will gaining some experience in a familiar geometry and 
code regime prior to tackling full task of parsing 
general GDML into  

Hmm : this is a bit of a cul-de-sac, as are not intending 
to support detdesc in general, however as have an existing 
node tree parsed from detdesc DYB PMT : it should be 
rather rapid to convert that into OpticksCSG tree 
serialization.

* this will allow to rapidly demonstrate OpticksCSG node 
  tree prior to tackling the larger job of converting GDML 
  into an "OpticksSceneGraph"




