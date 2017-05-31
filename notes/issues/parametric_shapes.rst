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
  


Facemask splitting
----------------------


Refining boundary faces
--------------------------

Subdivision of triangles along boolean join by generating 
more parametric triangles would be good...

* but add_vertex_unique uniquing stymies somewhat the labelling of vertices
  with the uv parameter and surface index values that gave rise to them ...
  because only the first vertex (based on its position) and its parameter
  values would get stored.

Given this complication, perhaps can use the implicit sdf for the 
sub-object to choose a face split location that is on the sub-object...
of directly jump to considering the compound sdf. And do the 
splits at the intersection boundary ?   

TriMeshT.hh::

    324   /** \brief Face split (= 1-to-3 split, calls corresponding PolyMeshT function).
    325    *
    326    * The properties of the new faces will be adjusted to the properties of the original face
    327    *
    328    * @param _fh Face handle that should be splitted
    329    * @param _p  New point position that will be inserted in the face
    330    *
    331    * @return Vertex handle of the new vertex
    332    */
    333   inline VertexHandle split_copy(FaceHandle _fh, const Point& _p)
    334   { const VertexHandle vh = this->add_vertex(_p);  PolyMesh::split_copy(_fh, vh); return vh; }
    335 


Boundary curve ?
-------------------


::

    simon:opticks blyth$ tboolean-;tboolean-hybrid-combinetest
    2017-05-31 20:52:53.167 INFO  [4234457] [main@70]  argc 2 argv[0] NOpenMeshCombineTest
    2017-05-31 20:52:53.168 INFO  [4234457] [>::build_parametric_primitive@823] NOpenMesh<T>::build_parametric ns 6 nu 16 nv 16 num_vert(raw) 1734 epsilon 1e-05
    2017-05-31 20:52:53.306 INFO  [4234457] [>::build_parametric_primitive@1006]  V 1538 E 4608 F 3072 Euler [(V - E + F)] 2
    2017-05-31 20:52:53.306 INFO  [4234457] [>::build_parametric_primitive@1007] build_parametric euler 2 expect_euler 2 EULER_OK  nvertices 1538 expect_nvertices 1538 NVERTICES_OK 
    2017-05-31 20:52:53.306 INFO  [4234457] [>::build_parametric_primitive@823] NOpenMesh<T>::build_parametric ns 1 nu 16 nv 16 num_vert(raw) 289 epsilon 1e-05
    2017-05-31 20:52:53.310 INFO  [4234457] [>::build_parametric_primitive@1006]  V 242 E 720 F 480 Euler [(V - E + F)] 2
    2017-05-31 20:52:53.310 INFO  [4234457] [>::build_parametric_primitive@1007] build_parametric euler 2 expect_euler 2 EULER_OK  nvertices 242 expect_nvertices 242 NVERTICES_OK 
    2017-05-31 20:52:53.310 INFO  [4234457] [>::build_parametric@97] build_parametric leftmesh 0x7f8829d018b0 rightmesh 0x7f8829d021d0
    2017-05-31 20:52:53.315 INFO  [4234457] [>::build_parametric@103] leftmesh inside node->right :   0 :   2728|  1 :     14|  2 :     14|  3 :     12|  4 :     28|  5 :     12|  6 :     12|  7 :    252|
    2017-05-31 20:52:53.315 INFO  [4234457] [>::build_parametric@106] rightmesh inside node->left :   0 :     72|  1 :     10|  2 :     10|  3 :     36|  4 :      8|  5 :     16|  6 :     16|  7 :    312|
    2017-05-31 20:52:53.874 INFO  [4234457] [>::dump_boundary_faces@168] boundary faces
    facemask:4
     heh 4862 vh 855 -> 837 uv (3; 5, 2) -> (3; 4, 1) a_pos vec3(200.000000, -75.000000, -150.000000) -> vec3(200.000000, -100.000000, -175.000000) _b_sdf          -4.744 ->          25.000
     heh 4765 vh 837 -> 838 uv (3; 4, 1) -> (3; 5, 1) a_pos vec3(200.000000, -100.000000, -175.000000) -> vec3(200.000000, -75.000000, -175.000000) _b_sdf          25.000 ->          15.058
     heh 4860 vh 838 -> 855 uv (3; 5, 1) -> (3; 5, 2) a_pos vec3(200.000000, -75.000000, -175.000000) -> vec3(200.000000, -75.000000, -150.000000) _b_sdf          15.058 ->          -4.744
         
     In this disposition can bisect along the lines of constant u and v...

                            (5,2)
                              + 
                             /|
                            / | 
                           /  | 
                          /   |
                         /    0
                        /     |
                       /      | 
                      +....0..-
                  (4,1)   (5,1)    


    facemask:1
     heh 4863 vh 837 -> 855 uv (3; 4, 1) -> (3; 5, 2) a_pos vec3(200.000000, -100.000000, -175.000000) -> vec3(200.000000, -75.000000, -150.000000) _b_sdf          25.000 ->          -4.744
     heh 4864 vh 855 -> 854 uv (3; 5, 2) -> (3; 4, 2) a_pos vec3(200.000000, -75.000000, -150.000000) -> vec3(200.000000, -100.000000, -150.000000) _b_sdf          -4.744 ->           6.155
     heh 4859 vh 854 -> 837 uv (3; 4, 2) -> (3; 4, 1) a_pos vec3(200.000000, -100.000000, -150.000000) -> vec3(200.000000, -100.000000, -175.000000) _b_sdf           6.155 ->          25.000


                


    facemask:4
     heh 4861 vh 855 -> 838 uv (3; 5, 2) -> (3; 5, 1) a_pos vec3(200.000000, -75.000000, -150.000000) -> vec3(200.000000, -75.000000, -175.000000) _b_sdf          -4.744 ->          15.058
     heh 4773 vh 838 -> 839 uv (3; 5, 1) -> (3; 6, 1) a_pos vec3(200.000000, -75.000000, -175.000000) -> vec3(200.000000, -50.000000, -175.000000) _b_sdf          15.058 ->           7.666
     heh 4866 vh 839 -> 855 uv (3; 6, 1) -> (3; 5, 2) a_pos vec3(200.000000, -50.000000, -175.000000) -> vec3(200.000000, -75.000000, -150.000000) _b_sdf           7.666 ->          -4.744
    facemask:3
     heh 4870 vh 839 -> 856 uv (3; 6, 1) -> (3; 6, 2) a_pos vec3(200.000000, -50.000000, -175.000000) -> vec3(200.000000, -50.000000, -150.000000) _b_sdf           7.666 ->         -12.917
     heh 4868 vh 856 -> 855 uv (3; 6, 2) -> (3; 5, 2) a_pos vec3(200.000000, -50.000000, -150.000000) -> vec3(200.000000, -75.000000, -150.000000) _b_sdf         -12.917 ->          -4.744
     heh 4867 vh 855 -> 839 uv (3; 5, 2) -> (3; 6, 1) a_pos vec3(200.000000, -75.000000, -150.000000) -> vec3(200.000000, -50.000000, -175.000000) _b_sdf          -4.744 ->           7.666
    facemask:4
     heh 4874 vh 857 -> 839 uv (3; 7, 2) -> (3; 6, 1) a_pos vec3(200.000000, -25.000000, -150.000000) -> vec3(200.000000, -50.000000, -175.000000) _b_sdf         -17.997 ->           7.666
     heh 4777 vh 839 -> 840 uv (3; 6, 1) -> (3; 7, 1) a_pos vec3(200.000000, -50.000000, -175.000000) -> vec3(200.000000, -25.000000, -175.000000) _b_sdf           7.666 ->           3.101
     heh 4872 vh 840 -> 857 uv (3; 7, 1) -> (3; 7, 2) a_pos vec3(200.000000, -25.000000, -175.000000) -> vec3(200.000000, -25.000000, -150.000000) _b_sdf           3.101 ->         -17.997
    facemask:3
     heh 4875 vh 839 -> 857 uv (3; 6, 1) -> (3; 7, 2) a_pos vec3(200.000000, -50.000000, -175.000000) -> vec3(200.000000, -25.000000, -150.000000) _b_sdf           7.666 ->         -17.997
     heh 4876 vh 857 -> 856 uv (3; 7, 2) -> (3; 6, 2) a_pos vec3(200.000000, -25.000000, -150.000000) -> vec3(200.000000, -50.000000, -150.000000) _b_sdf         -17.997 ->         -12.917
     heh 4871 vh 856 -> 839 uv (3; 6, 2) -> (3; 6, 1) a_pos vec3(200.000000, -50.000000, -150.000000) -> vec3(200.000000, -50.000000, -175.000000) _b_sdf         -12.917 ->           7.666
    facemask:4
     heh 4873 vh 857 -> 840 uv (3; 7, 2) -> (3; 7, 1) a_pos vec3(200.000000, -25.000000, -150.000000) -> vec3(200.000000, -25.000000, -175.000000) _b_sdf         -17.997 ->           3.101
     heh 4785 vh 840 -> 841 uv (3; 7, 1) -> (3; 8, 1) a_pos vec3(200.000000, -25.000000, -175.000000) -> vec3(200.000000, 0.000000, -175.000000) _b_sdf           3.101 ->           1.556
     heh 4878 vh 841 -> 857 uv (3; 8, 1) -> (3; 7, 2) a_pos vec3(200.000000, 0.000000, -175.000000) -> vec3(200.000000, -25.000000, -150.000000) _b_sdf           1.556 ->         -17.997
    facemask:3
     heh 4882 vh 841 -> 858 uv (3; 8, 1) -> (3; 8, 2) a_pos vec3(200.000000, 0.000000, -175.000000) -> vec3(200.000000, 0.000000, -150.000000) _b_sdf           1.556 ->         -19.722
     heh 4880 vh 858 -> 857 uv (3; 8, 2) -> (3; 7, 2) a_pos vec3(200.000000, 0.000000, -150.000000) -> vec3(200.000000, -25.000000, -150.000000) _b_sdf         -19.722 ->         -17.997
     heh 4879 vh 857 -> 841 uv (3; 7, 2) -> (3; 8, 1) a_pos vec3(200.000000, -25.000000, -150.000000) -> vec3(200.000000, 0.000000, -175.000000) _b_sdf         -17.997 ->           1.556
    facemask:4
     heh 4886 vh 859 -> 841 uv (3; 9, 2) -> (3; 8, 1) a_pos vec3(200.000000, 25.000000, -150.000000) -> vec3(200.000000, 0.000000, -175.000000) _b_sdf         -17.997 ->           1.556
     heh 4789 vh 841 -> 842 uv (3; 8, 1) -> (3; 9, 1) a_pos vec3(200.000000, 0.000000, -175.000000) -> vec3(200.000000, 25.000000, -175.000000) _b_sdf           1.556 ->           3.101
     heh 4884 vh 842 -> 859 uv (3; 9, 1) -> (3; 9, 2) a_pos vec3(200.000000, 25.000000, -175.000000) -> vec3(200.000000, 25.000000, -150.000000) _b_sdf           3.101 ->         -17.997
    facemask:3
     heh 4887 vh 841 -> 859 uv (3; 8, 1) -> (3; 9, 2) a_pos vec3(200.000000, 0.000000, -175.000000) -> vec3(200.000000, 25.000000, -150.000000) _b_sdf           1.556 ->         -17.997
     heh 4888 vh 859 -> 858 uv (3; 9, 2) -> (3; 8, 2) a_pos vec3(200.000000, 25.000000, -150.000000) -> vec3(200.000000, 0.000000, -150.000000) _b_sdf         -17.997 ->         -19.722
     heh 4883 vh 858 -> 841 uv (3; 8, 2) -> (3; 8, 1) a_pos vec3(200.000000, 0.000000, -150.000000) -> vec3(200.000000, 0.000000, -175.000000) _b_sdf         -19.722 ->           1.556
    facemask:4
     heh 4885 vh 859 -> 842 uv (3; 9, 2) -> (3; 9, 1) a_pos vec3(200.000000, 25.000000, -150.000000) -> vec3(200.000000, 25.000000, -175.000000) _b_sdf         -17.997 ->           3.101
     heh 4797 vh 842 -> 843 uv (3; 9, 1) -> (3;10, 1) a_pos vec3(200.000000, 25.000000, -175.000000) -> vec3(200.000000, 50.000000, -175.000000) _b_sdf           3.101 ->           7.666
     heh 4890 vh 843 -> 859 uv (3;10, 1) -> (3; 9, 2) a_pos vec3(200.000000, 50.000000, -175.000000) -> vec3(200.000000, 25.000000, -150.000000) _b_sdf           7.666 ->         -17.997






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



