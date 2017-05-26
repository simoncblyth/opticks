Meshes from Parametric Shapes
===============================

* Complete sphere is easy as its a single continuous sheet
  but what about cylinders with endcaps, chopped spheres, 
  chopped cones, cube or general convex polyhedron defined by 
  planes. 

* Endcap disks need control points to triangle fan around. 

* How to join the sheets together... 

* Each 3D vertex needs to have associated parametric uv (0:1,0:1), 
  also perhaps a sheet index (eg for a cube).  

* normals

* jump direct into OpenMesh ?

* need subdivision 


Review Parametric Approaches
---------------------------------

par
~~~


* https://github.com/prideout/par/blob/master/par_shapes.h
* http://github.prideout.net/shapes

Uses merge and weld approach to handle multi-sheet.



