Review Progress and Where Next ?
===================================

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

* Very small number of primitives: sphere, box

* Interactive composited visualization of the ray traced images and rasterized 
  polygonization meshes together with photon propagations thru the geometry 
  for convenient geometry debugging.


TODO: add more primitives
---------------------------

* cylinder (infinite, truncated)
* cone
* polycone
* tubs (ie cylinder with thickness)

There is some choice over where to
draw the line between primitives and compound CSG trees, 
eg a cylinder with thickness could be handles by 
a CSG difference ? 

GPU performance is so good anyhow, expect should
be able to have a much smaller number of primitives
and make use of the general transforms (translate-rotate-scale)
and CSG node operations to mimick G4 "primitives". 

Could implement facade primitives that are internally compound, 
if that would be convenient.

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


What is NOT needed for each primitive ?
-------------------------------------------

* Do not need a parametrized surface description to generate a mesh,
  this is due to the use of generalized polygonization isosurface extraction 
  using signed distance functions which can be CSG combined with min/max. 


TODO : Push CSG node tree support thru to cfg4
------------------------------------------------

Add:

* creation of Geant4 geometries from the CSG node tree description.
* comparisons of GPU and CPU propagations using CSG node tree geometries


TODO: Apply CSG tree model to analytic PMT 
-------------------------------------------

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


TODO: Converting GDML description into instance-ized CSG trees "OpticksSceneGraph"
-----------------------------------------------------------------------------------
   
CSG node trees are intended to describe individual "solids"
not entire scenes.  These need to be combines into
an OpticksSceneGraph format/serialization.

This is similar to the conversion of G4DAE/COLLADA trees 
into GPU geometries. But as starting from source GDML tree, 
can do a more complete job.

* use instancing for *all* solids (ie for all distinct shapes)
  minimizing the GPU memory requirements
  
  * ggeo analyses the G4DAE node tree to find
    repeated geometry ... this works but when have 
    direct access to the source GDML tree presumably 
    can do better by directly accessing all distinct shapes, 
    making CSG trees for each of them 

  * unsure how good GDML is at avoiding repetion, suspect 
    that some digesting will be needed 

  * polygonize the CSG trees into meshes, serialize and
    persist them together with the source CSG trees

    Currently with test geometry the meshes are not 
    persisted, just directly uploaded to GPU/OpenGL, but 
    when handling full geometries need to work with 
    a geocache serialization to avoid repeating work.

* construct scene graph structure (and serialization)
  aggregating references to the csg tree instances 
  together with their transforms

  * review OptiX geometry handling and OpenGL instancing, as currently 
    used to see how best to structure this to be 
    easily uploaded to GPU 


Whats needed in OpticksSceneGraph ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple buffer layout, for GPU consumption, be guided by customers:

* OptiX geometry instancing
* OpenGL geometry instancing

For each instance (perhaps uint4 buffer)

* unsigned index reference to CSG tree,  
* unsigned index reference to transform 
* identity code or reference to identity  

What to do different from current GGeo ?

* GGeo is mesh-centric, aim for instance-centric 
* design with simple serialization directory layout in mind 
* defer concatenation into big buffers as late as possible,
  retaining structure in directories for easy debug 


GDML->GGeo vs G4DAE->GGeo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So the process of converting GDML description, needs to 
follow a very similar course to the conversion of G4DAE 
COLLADA into a GPU description (GGeo and OGeo).

Do this inside GGeo ? Or another package ?

* initially start in GGeo and see how it goes
* recall GGeo was intended as a dumb substrate initially ...

The tasks are the same, so regard it as improving GGeo, 
not doing something new.


Validation
~~~~~~~~~~~

* implement in cfg4- OpticksSceneGraph -> G4 conversion, so 
  can compare two routes for geometry 

  * GDML -> G4 
  * GDML -> OpticksSceneGraph -> G4   


OpticksSceneGraph Technicalites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* use python for parsing GDML rather than working in C++ with the G4 parse ? 
  Then can start from the (pmt-) dd.py detdesc/lxml parse 
  and bring it over to work with GDML 
    
* no reason why not to use python for input geometry conversion, 
  as in production this is only done once for each geometry 

Multi-level approach similar to NCSG chain, perhaps steered with 
an "NScene" ?  

* python prepares input serialization from the GDML, 
  finding all distinct shapes and writing CSG tree serializations
  for them,  
  (directory structure of .npy .json .txt)

* npy- embellishes the directory structure 
  eg using NPolygonization to write meshes into directory tree

* ggeo-  intermediate GPU geometry prep, however
  as have more control over NScene than with the COLLADA/Assimp/GGeo
  route expect will need less action at GGeo level  

* oglrap- to OpenGL

* ogeo-  to OptiX


Why not parse with G4 and work with G4 in-memory tree ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* prefer to keep G4 dependency to a minimum, yields more generally usable code
* promotes an independent approach 
* avoids having to work with G4 too much 


TODO: Add NPolygonization of partlist ?
--------------------------------------------

World allow cleaning up the currently dirty GMaker/PmtInBox mode, 
which makes the adhoc association of a loaded PMT mesh 
with analytic part list.  

Would need to add solids: cylinder

YES BUT, partlist are very limited, only keep them around as 
a possible optimization of csg tree, so this is too much of a cul-de-sac.






