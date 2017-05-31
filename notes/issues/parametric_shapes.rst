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


Joining Endcaps to body 
--------------------------

* when control the parametric domains and stepping 
  can ensure that get identical vertices at borders of 
  the uv spaces, which can then 
  simply be collaped to one in combined meshes 

  * actually just using add_vertex unique with 
    some suitable length epsilon below which to snap
    vertices together this turns out to be straightforward.

 
  * problem comes with general boolean combinations where there will
    not be equal numbers of vertices around the boundaries that 
    need to be stitched together : need to come up with a local edge split/join 
    algorithm that can locally change vertex counts around the boundary region 
  





Objective of parametric geometry description
---------------------------------------------

Aiming to create a hybrid implicit/parametric polygonization
of general CSG binary tree defined solids.  For this will 
need to:

* find intersection contours (using implicit sdf)
* split each shape according to whether the verts are inside the other shape, 
  using parametric approach to find the intersect of the shapes
* will need subdiv when too few verts 
* weld the split shapes together 

The welding of cut shapes is similar to the problem of joining endcaps 
to other sheets.

Hmm for cube (and other convex polyhedra defined by planes) 
the welding will need to handle same verts from multiple sheets (4 for cube). 



Welding with OpenMesh ?
--------------------------

Joining meshes
~~~~~~~~~~~~~~~~

https://mailman.rwth-aachen.de/pipermail/openmesh/2010-March/000405.html

there is no predefined function to do that. 

The easiest way to do it is to create a map while adding the vertices to the 
other mesh, mapping from the old mesh vertex handle to the new mesh vertex 
handle. Than you can just iterate over all faces of the old mesh, use a 
FaceVertex iterator and add the face with the mapped vertex handles to the new 
mesh.

Best Regards,
Jan Möbius

On Samstag, 6. March 2010, John Terrell wrote:
> Hi everyone, I have a need to combine multiple meshes into a single mesh
>  that contains all the data of the originals.   It's easy to copy vertex
>  info over (just iterating over the vertices and adding them to the
>  aggregate mesh) but I can't figure out a nice way to add the face data
>  (perhaps I'm missing something trivial).    Any help?
> 
> Thanks.
> 
> -John



Review Parametric Approaches
---------------------------------

par
~~~


* https://github.com/prideout/par/blob/master/par_shapes.h
* http://github.prideout.net/shapes

Uses merge and weld approach to handle multi-sheet.



