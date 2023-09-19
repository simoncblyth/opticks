DONE : X4Mesh_poly_plotting_worth_doing_similar_in_new_workflow
====================================================================

HMM for debugging complex geometry its useful to 
have a polydata conversion of a solid as was done by::

   X4MeshTest

See http://localhost/env/presentation/opticks_20211223_pre_xmas.html 


Modern Take on X4MeshTest ?
------------------------------

* creation uses G4Polyhedron but the focus is the output : so U4Mesh.h ?  
* U4Solid::polygonize that uses U4Mesh : keep testing
* no need for GLTF complications : use pyvista for viz
* persisting can use NPFold : try to avoid dedicated sysrap type


DONE : With U4Mesh.h
-----------------------

::

	u4/U4Mesh.h
	u4/tests/U4Mesh_test.cc
	u4/tests/U4Mesh_test.py
	u4/tests/U4Mesh_test.sh


