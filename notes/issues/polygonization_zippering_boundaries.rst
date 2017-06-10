Polygonization Zippering Boundaries
======================================

Zippering Realizations for CSG union of box and sphere
---------------------------------------------------------

* left and right mesh frontier "ribbons" of tris, each ribbon having inner/outer boundary edge loops

* outer loop from leftmesh needs to be zippered with the outer loop of the rightmesh frontier ribbon equivalent

* inner loops can be discarded, tris can be discarded, 

  * hmm NO : can bisect frontier crossing edges to get verts on the frontier "ring" 

* ONLY NEED TO RETAIN outer loop vertices from left and right 

* only the verts from one side of the ribbons need to be 


::


      . . . . . .*. . . . . . . . . . . * . . . .                     (rightmesh inner loop, can be discarded)
                / \
               /   \
      --------/-----\-------------+-----------+---     leftmesh outer loop
             /       \             \         /
       -  - /-  -   - \-  -  -  -  -\ -  -  /  -  -  -  -  -
           /           \             \     /
      ----+-------------+-------------\---/--------    rightmesh outer loop
                                       \ / 
        . . . . . . . . . . . . . . . . * . . . .                      (left mesh inner loop, can be discarded)




OpenFlipper Boundary Snap
----------------------------

OpenFlipper does something similar, 

* /usr/local/opticks/externals/openflipper/OpenFlipper-3.1/Plugin-MeshRepair/BoundarySnappingT.cc

An un-obfuscated version using typedefs

* /Users/blyth/opticks/opticksnpy/tests/BoundarySnappingT.cc



Mesh zippering
-----------------

* boils down to adding faces that join the 
* problem is they are not 1-1, will be more vertices  
* constraint : cannot add vertices to the loops


::


      ----------+-----------------+-----------+---     leftmesh outer loop
                                              
       -  -  -  -   -  -  -  -  -  -  -  -     -  -    actual analytic interface between sub-objects
                                            
      ----+-------------+--------------+------------    rightmesh outer loop
                                           


* for each vert find the nearest one on the other side 
* find the nearest vert on o

* hmm the halfedge loop should give the correct ordering of each loop, its just a registration
  issue of lining them up and using triangle fans to deal with too many verts 


* another possibility is approximate the analytic boundary and add a line of vertices
  along it could do sdf evaluation, actually this 

  * would be helpful as are free to put as many verts as like along the analytic curve,
    can help with vertex count mismatch ... then have a 


::


      ----------+-----------------+-----------+---     leftmesh outer loop
                                              
       -  +     +    +     +    +      +    +    +     analytic interface verts added flexibly 
                                            
      ----+-------------+--------------+------------    rightmesh outer loop
 





Investigate boundary loops with modded combine_hybrid
-------------------------------------------------------

* modification just copies frontier faces from leftmesh 

::

   tboolean-;tboolean-hybrid

   ## visualization shows "pixelated" ring of tris
   ## approximating the box-sphere intersection circle  

::

    133 template <typename T>
    134 void NOpenMesh<T>::build_csg()
    135 {

    226 template <typename T>
    227 void NOpenMesh<T>::combine_hybrid( )
    228 {

    300 
    301     if(node->type == CSG_UNION)
    302     {
    303         //NOpenMeshPropType prop = PROP_OUTSIDE_OTHER ;
    304         NOpenMeshPropType prop = PROP_FRONTIER ;
    305 
    306         build.copy_faces( leftmesh,  prop, epsilon );
    307         //build.copy_faces( rightmesh, prop, epsilon );
    308 
    309         //subdiv.sqrt3_refine( FIND_ALL_FACE , -1 );  // test refining copied over frontier tris
    310 
    311     }
    ...
    326     int nloop = find.find_boundary_loops() ;
    327     // hmm expecting 2, but thats geometry specific
    328 
    329     find.dump_boundary_loops("find.dump_boundary_loops", true );
    330 



Leftmesh frontier "ribbon" with outer and inner boundary edge loops
------------------------------------------------------------------------

* the outer loop is the one that needs to be zippered with the outer boundary loop of the rightmesh frontier ribbon equivalent

::

    2017-06-09 19:14:21.233 INFO  [6274680] [>::combine_hybrid@247] combine_hybrid leftmesh 0x7fa0d5adedc0 rightmesh 0x7fa0d5adf900
    2017-06-09 19:14:21.239 INFO  [6274680] [>::mark_faces@401] mark_faces 0:  2728|1:    14|2:    14|3:    12|4:    28|5:    12|6:    12|7:   252|
    2017-06-09 19:14:21.239 INFO  [6274680] [>::mark_faces@401] mark_faces 0:    68|1:    10|2:    10|3:     8|4:     4|5:    12|6:    12|7:   356|
    2017-06-09 19:14:21.242 INFO  [6274680] [>::dump_boundary_loops@528] find.dump_boundary_loops
     nloop 2
    NOpenMeshBoundary::desc halfedge boundary loop  index 1 start 1 num_heh 56 :  1 9 67 75 91 99 115 123 141 161 173 193 205 225 237 239 261 263 285 287...
    NOpenMeshBoundary::desc halfedge boundary loop  index 2 start 15 num_heh 36 :  15 25 31 41 47 57 81 105 129 153 165 185 197 217 229 249 273 297 355 347...
    2017-06-09 19:14:21.242 INFO  [6274680] [>::dump@78] NOpenMeshBoundary::dump halfedge boundary loop  index 1 start 1 num_heh 56 :  1 9 67 75 91 99 115 123 141 161 173 193 205 225 237 239 261 263 285 287...

    //
    //   dumping the "to" vertices around 1st heh loop  
    //
    //   all verts are on the wall of the box (left sub-object) ... hence sdf_C and sdf_L are zero (on- isosurface)
    //
    //   sdf_R all small positive 
    //         boundary is fully outside the right sub-object sphere 
    //
    //     THIS OUTER BOUNDARY IS THE ONE THAT NEEDS TO BE ZIPPERED ONTO THE OTHER MESH
    //
    //
     i    0 heh    1 eh    0 tv    0 (        201.000,       -100.500,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R      26.35
     i    1 heh    9 eh    4 tv    3 (        201.000,       -100.500,       -150.750)  sdf_C       0.00 sdf_L       0.00 sdf_R       7.43
     i    2 heh   67 eh   33 tv   18 (        201.000,       -125.625,       -150.750)  sdf_C       0.00 sdf_L       0.00 sdf_R      20.70
     i    3 heh   75 eh   37 tv   20 (        201.000,       -125.625,       -125.625)  sdf_C       0.00 sdf_L       0.00 sdf_R       4.36
     i    4 heh   91 eh   45 tv   24 (        201.000,       -150.750,       -125.625)  sdf_C       0.00 sdf_L       0.00 sdf_R      20.70
     i    5 heh   99 eh   49 tv   26 (        201.000,       -150.750,       -100.500)  sdf_C       0.00 sdf_L       0.00 sdf_R       7.43
     i    6 heh  115 eh   57 tv   30 (        201.000,       -175.875,       -100.500)  sdf_C       0.00 sdf_L       0.00 sdf_R      26.35
     i    7 heh  123 eh   61 tv   32 (        201.000,       -175.875,        -75.375)  sdf_C       0.00 sdf_L       0.00 sdf_R      16.37
     ...
     i   47 heh   83 eh   41 tv   17 (        201.000,        100.500,       -150.750)  sdf_C       0.00 sdf_L       0.00 sdf_R       7.43
     i   48 heh   65 eh   32 tv   16 (        201.000,        100.500,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R      26.35
     i   49 heh   59 eh   29 tv   14 (        201.000,         75.375,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R      16.37
     i   50 heh   51 eh   25 tv   12 (        201.000,         50.250,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R       8.95
     i   51 heh   43 eh   21 tv   10 (        201.000,         25.125,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R       4.36
     i   52 heh   35 eh   17 tv    8 (        201.000,          0.000,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R       2.81
     i   53 heh   27 eh   13 tv    6 (        201.000,        -25.125,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R       4.36
     i   54 heh   19 eh    9 tv    4 (        201.000,        -50.250,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R       8.95
     i   55 heh   11 eh    5 tv    1 (        201.000,        -75.375,       -175.875)  sdf_C       0.00 sdf_L       0.00 sdf_R      16.37
    2017-06-09 19:14:21.243 INFO  [6274680] [>::dump@78] NOpenMeshBoundary::dump halfedge boundary loop  index 2 start 15 num_heh 36 :  15 25 31 41 47 57 81 105 129 153 165 185 197 217 229 249 273 297 355 347...

     /// boundary loop on other side of the ribbon 
     ///      again on 
     ///
     ///    
     ///    sdf_R all negative 
     ///            boundary is fully inside the sphere
     ///
     ///    sdf_C all negative (same as sdf_R)
     ///            boundary is fully inside the union of sphere and box
     ///           

     i    0 heh   15 eh    7 tv    5 (        201.000,        -50.250,       -150.750)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i    1 heh   25 eh   12 tv    7 (        201.000,        -25.125,       -150.750)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i    2 heh   31 eh   15 tv    9 (        201.000,          0.000,       -150.750)  sdf_C     -18.54 sdf_L       0.00 sdf_R     -18.54
     i    3 heh   41 eh   20 tv   11 (        201.000,         25.125,       -150.750)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i    4 heh   47 eh   23 tv   13 (        201.000,         50.250,       -150.750)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i    5 heh   57 eh   28 tv   15 (        201.000,         75.375,       -150.750)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i    6 heh   81 eh   40 tv   21 (        201.000,        100.500,       -125.625)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i    7 heh  105 eh   52 tv   27 (        201.000,        125.625,       -100.500)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i    8 heh  129 eh   64 tv   33 (        201.000,        150.750,        -75.375)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i    9 heh  153 eh   76 tv   39 (        201.000,        150.750,        -50.250)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   10 heh  165 eh   82 tv   42 (        201.000,        150.750,        -25.125)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   11 heh  185 eh   92 tv   47 (        201.000,        150.750,          0.000)  sdf_C     -18.54 sdf_L       0.00 sdf_R     -18.54
     i   12 heh  197 eh   98 tv   50 (        201.000,        150.750,         25.125)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   13 heh  217 eh  108 tv   55 (        201.000,        150.750,         50.250)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   14 heh  229 eh  114 tv   58 (        201.000,        150.750,         75.375)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i   15 heh  249 eh  124 tv   64 (        201.000,        125.625,        100.500)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   16 heh  273 eh  136 tv   70 (        201.000,        100.500,        125.625)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   17 heh  297 eh  148 tv   76 (        201.000,         75.375,        150.750)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i   18 heh  355 eh  177 tv   88 (        201.000,         50.250,        150.750)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   19 heh  347 eh  173 tv   86 (        201.000,         25.125,        150.750)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   20 heh  339 eh  169 tv   84 (        201.000,          0.000,        150.750)  sdf_C     -18.54 sdf_L       0.00 sdf_R     -18.54
     i   21 heh  331 eh  165 tv   82 (        201.000,        -25.125,        150.750)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   22 heh  323 eh  161 tv   80 (        201.000,        -50.250,        150.750)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   23 heh  315 eh  157 tv   74 (        201.000,        -75.375,        150.750)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i   24 heh  293 eh  146 tv   68 (        201.000,       -100.500,        125.625)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   25 heh  269 eh  134 tv   62 (        201.000,       -125.625,        100.500)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   26 heh  245 eh  122 tv   56 (        201.000,       -150.750,         75.375)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i   27 heh  219 eh  109 tv   53 (        201.000,       -150.750,         50.250)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   28 heh  209 eh  104 tv   48 (        201.000,       -150.750,         25.125)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   29 heh  187 eh   93 tv   45 (        201.000,       -150.750,          0.000)  sdf_C     -18.54 sdf_L       0.00 sdf_R     -18.54
     i   30 heh  177 eh   88 tv   40 (        201.000,       -150.750,        -25.125)  sdf_C     -16.81 sdf_L       0.00 sdf_R     -16.81
     i   31 heh  155 eh   77 tv   37 (        201.000,       -150.750,        -50.250)  sdf_C     -11.71 sdf_L       0.00 sdf_R     -11.71
     i   32 heh  145 eh   72 tv   31 (        201.000,       -150.750,        -75.375)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
     i   33 heh  125 eh   62 tv   25 (        201.000,       -125.625,       -100.500)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   34 heh  101 eh   50 tv   19 (        201.000,       -100.500,       -125.625)  sdf_C     -10.05 sdf_L       0.00 sdf_R     -10.05
     i   35 heh   77 eh   38 tv    2 (        201.000,        -75.375,       -150.750)  sdf_C      -3.51 sdf_L       0.00 sdf_R      -3.51
    2017-06-09 19:14:21.244 INFO  [6274680] [>::combine_hybrid@333] combine_hybrid boundary_loops 2
    2017-06-09 19:14:21.244 INFO  [6274680] [>::build_csg@205] NOpenMesh<T>::build_csg DONE 
     leftmesh   V 1538 E 4608 F 3072 Euler [(V - E + F)] 2
     rightmesh  V 242 E 720 F 480 Euler [(V - E + F)] 2
     combined  V 92 E 184 F 92 Euler [(V - E + F)] 0


