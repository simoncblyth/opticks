Geometry Testing (Feb 2022)
============================


I welcome all work that leads to improvements to Opticks correctness and performance.

Making such improvements however demands familiarity with the details of the 
Opticks and Geant4 geometry implementations (especially CSG) and the translation 
between them. Given this background : creation of a test suite that creates a 
large variety of Geant4 solids is a good idea as it can be helpful without 
requiring great familiarity, initially anyhow.

It is important to be aware that Opticks transition to the all new NVIDIA OptiX 7 API 
is not complete and the default OptiX used by the Opticks build is still < 7.
Clearly it would not be productive to validate or optimize the pre-7 workflow 
as the future is with the all new API. That means lots of code, eg all of OptiXRap,
is about to be retired. It is difficult to predict exactly when, but within months I 
aim to make the leap to OptiX 7. The source of delay is other urgent JUNO work. 

Opticks has a great deal of geometry testing functionality already.
The relevant packages and sources for testing Opticks geometry with the OptiX 7 API 
are listed below.  Anyone wishing to test/improve/extend the 
Opticks geometry implementation shoud get familiar with the below in order to avoid
reinventing the wheel. 


extg4/X4SolidMaker.cc 
   source of test Geant4 solids 
   const G4VSolid* X4SolidMaker::Make(const char* qname)

extg4/xxs.sh 
   creates 2D cross sections of Geant4 solids providing 
   a good way to find problems such as spurious intersects 
   at geant4 level

extg4/X4MeshTest.sh 
   creates 3D visualizations of Geant4 polygonized meshes using pyvista

GeoChain 
   performs full chain of geometry conversions, taking a Geant4 solid 
   and converting it via npy/NNode and GGeo into CSG/CSGFoundry  

GeoChain/run.sh
   invokes the conversion starting with X4SolidMaker::Make

CSG_GGeo
   converts GGeo geometries into the CSG model, used by GeoChain

CSG
   OptiX 7 (and <7) compatible geometry model  
      
CSG/csg_geochain.sh
    CPU test of the CUDA compatible CSG implementation : csg_intersect_tree.h csg_intersect_node.h  
    This loads GeoChain converted CSG geometries and shoots rays at them, 
    saving first intersects into .npy files for python plotting using pyvista and matplotlib 

CSG/sdf_geochain.sh 
    saves "SDF" distance fields of a solid and uses pyvista iso-surface finding (eg marching cubes) 
    to present a 3D render of the surface.
    *distance_node functions are currently not implemented for all primitives* 
    
    The distance_node functions return distance to surface for any 3D position, 
    as that should be close to zero for all intersects onto the solid this
    provides a useful way of finding some forms of spurious intersect problems.  

qudarap
   CUDA elements of the simulation not requiring intersections with geometry
   such as photon generation, this is used from CSGOptiX 

CSGOptiX
   Uses OptiX 7 to render and simulate within 
   Unlike all the above packages this one does depends on OptiX 7, 
  
   CAUTION : CSGOptiX builds by default against OptiX < 7, to build against OptiX 7 
   follow the pasted opticks-build7-notes in the postscript below.

CSGOptiX/cxr_geochain.sh 
    Creates 3D renders of CSG geometries converted via GeoChain

CSGOptiX/cxs_geochain.sh 
    Creates 2D cross-sections of CSG geometries composed of intersections 
    onto the geometry. 

    * not implemented for OptiX < 7 (only works with OptiX 7)



Opticks is not something that can be picked up quickly, or installed quickly either. 
For hackathons or similar to be productive requires significant preparation work
by everyone that will be working with Opticks.



List of GEOM scripts
-----------------------
 
All the below scripts access the $HOME/.opticks/GEOM.txt file or GEOM envvar to configure the geometry to create, visualize or shoot single rays at.
The first three scripts run directly from the G4VSolid or G4PhysicalVolume. 
The last four scripts require the third translate.sh or fourth CSGMakerTest.sh  script to have been run first:: 


     x4 ; ./X4MeshTest.sh    ## CPU : Geant4 polygons visualized with pyvista

     x4 ; ./xxs.sh           ## CPU : 2D Geant4 intersects visualized with matplotlib and/or pyvista



     gc ; ./translate.sh     ## CPU : Create CSGFoundry geometry via translation of Geant4 geometry via the GeoChain 

     c ; ./CSGMakerTest.sh   ## CPU : Create CSGFoundry geometry directly at CSGNode/CSGPrim/CSGSolid level with CSGMaker 



     c ; ./sdf_geochain.sh   ## CPU : 3D Opticks distance field visualised with pyvista iso-surface finding 

     c ; ./csg_geochain.sh   ## CPU : 2D(or 3D) pyvista visualization of Opticks intersects (CPU test run of CUDA comparible intersect code)

     c ; ./CSGQueryTest.sh   ## CPU : test mostly used for shooting single rays at geometry, useful after compiling with DEBUG flag enabled   




     cx ; ./cxr_geochain.sh  ## GPU : 3D OptiX/Opticks render of geometry      




