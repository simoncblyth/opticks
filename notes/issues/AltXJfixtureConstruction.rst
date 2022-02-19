AltXJfixtureConstruction
===========================

Looks like Z-shift transforms present in G4VSolid are not getting thru the GeoChain:: 

     geom  ## check that only one entry is uncommented in GEOM.txt eg AltXJfixtureConstruction_YZ


     x4 ; ./X4MeshTest.sh    ## CPU : Geant4 polygons visualized with pyvista

     x4 ; ./xxs.sh           ## CPU : 2D intersects visualized with matplotlib and/or pyvista

     c ; ./sdf_geochain.sh   ## CPU : 3D distance field visualised with pyvista iso-surface finding 

     c ; ./csg_geochain.sh   ## CPU : 2D (or 3D) pyvista visualization of intersects (CPU test run of CUDA comparible intersect code)

     cx ; ./cxr_geochain.sh  ## GPU : 3D OptiX render of geometry      





