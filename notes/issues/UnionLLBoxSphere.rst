UnionLLBoxSphere
===================


SPURIOUS=1 IXYZ=0,0,0 ./csg_geochain.sh ana 

See ascii art 

CSGMaker::makeUnionLLBoxSphere
--------------------------------

radius=100.f fullside=100.f


                                   (-50,100)                  (50,100)
                                    +                           +
                          .                 .             .            .
                                                   +  
                     .                                                       .
                                              .           . 
 
                  .                                                                 .
                                         .

              .                       .                       \                        .
             
                               
           -|-----------------------s0-------------O------------|s1--------------------|-------------
                              (-50,0,0)                       (50,0,0)                        

              \                        \                      /                       /


                   
                 
                                                    +


                                    +                           +
                                  (-50,-100)                 (50, -100)


            |          |            |             |            |           |          |
          -150        -100         -50            0            50         100        150
          



Shoting from (0,0,0) to right get internal hit


::

    epsilon:CSG blyth$ ./CSGQueryTest.sh 
    === ./CSGQueryTest.sh catgeom UnionLLBoxSphere_XY override of default geom
    === ./CSGQueryTest.sh catgeom UnionLLBoxSphere_XY geom UnionLLBoxSphere GEOM UnionLLBoxSphere
                               One HIT
                        q0 norm t (   -1.0000   -0.0000   -0.0000   50.0000)
                       q1 ipos sd (   50.0000    0.0000    0.0000  100.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (    0.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )


    epsilon:CSG blyth$ ./CSGQueryTest.sh 
    === ./CSGQueryTest.sh catgeom UnionLLBoxSphere_XY override of default geom
    === ./CSGQueryTest.sh catgeom UnionLLBoxSphere_XY geom UnionLLBoxSphere GEOM UnionLLBoxSphere
    //intersect_prim typecode 4 name difference 
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name contiguous depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 node_or_leaf 1 
    //intersect_node typecode 11 name contiguous 
    //intersect_node_contiguous 
    //distance_leaf typecode 12 name discontiguous complement 0 sd     0.0000 

    ^^^^^^^^ thats wrong this should only see leaf nodes 


    //distance_node_list isub 0 sub_sd     0.0000 sd     0.0000 
    //distance_leaf typecode 101 name sphere complement 0 sd   -50.0000 
    //distance_node_list isub 1 sub_sd   -50.0000 sd   -50.0000 
    //intersect_node_contiguous sd   -50.0000 inside_or_surface 1 num_sub 2 offset_sub 3 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (    0.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     1.0000     0.0000     0.0000    50.0000)  
    //intersect_leaf valid_isect 1 isect (    1.0000     0.0000     0.0000    50.0000)   
    //intersect_leaf typecode 101 gtransformIdx 2 
    //intersect_leaf ray_origin (    0.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     1.0000     0.0000     0.0000   150.0000)  
    //intersect_leaf valid_isect 1 isect (    1.0000     0.0000     0.0000   150.0000)   




Fixed one bug, but still getting early HIT::

     47 INTERSECT_FUNC
     48 float distance_node_list( unsigned typecode, const float3& pos, const CSGNode* node, const float4* plan, const qat4* itra )
     49 {
     50     const unsigned num_sub = node->subNum() ;
     51     float sd = typecode == CSG_OVERLAP ? -RT_DEFAULT_MAX : RT_DEFAULT_MAX ;
     52     for(unsigned isub=0 ; isub < num_sub ; isub++)
     53     {
     54          const CSGNode* sub_node = node+1u+isub ;  
     55          // TOFIX: this is assuming the sub_node follow the node, which they do not for lists within trees
     56          



Still loadsa internals::

    IXYZ=0,0,0 ./csg_geochain.sh ana



