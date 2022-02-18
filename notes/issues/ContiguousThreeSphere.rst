ContiguousThreeSphere
========================


./sdf_geochain.sh 
    geometry via distance functions is as expected, all three spheres visible

./csg_geochain.sh 
    Third sphere not appearing, but temporary switching to 'D' for Discontiguous shows that
    it is there and it overlaps both the others. 

SPURIOUS=1 IXYZ=0,0,0 ./csg_geochain.sh ana 
    spurious intersects are where the 3rd sphere should be 

SPURIOUS=1 IXYZ=0,-5,0 ./csg_geochain.sh ana 
    no intersects from within the third sphere 


CSGMaker::makeDiscontiguousTwoSphere
---------------------------------------

Have suspicion about inverse transforms : so try to distinguish using non-symmetric pair in DiscontiguousTwoSphere

    ./CSGMakerTest.sh 


Checking the sense of the transforms : get the expected ones.::

    .
                   Y
    (-100,200)     |
    (-side, 2*side)|
         t1        |
           +       |
                   |
                   |
                   |      +   t0 (side, 0.5*side)    (100,50)
                   |                            
                   |
    ---=-----------O------------------------------ X




Tee up a MISS that should be a HIT in +X direction from origin of third sphere::

    epsilon:CSG blyth$ ./CSGQueryTest.sh 
    === ./CSGQueryTest.sh catgeom ContiguousThreeSphere_XY override of default geom
    === ./CSGQueryTest.sh catgeom ContiguousThreeSphere_XY geom ContiguousThreeSphere GEOM ContiguousThreeSphere
                               One MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (    0.0000  -70.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )


That reveals the cause is subOffset 0 which should be 1, due to CSGMaker::makeList not being updated
for more general sub referencing::

    epsilon:CSG blyth$ ./CSGQueryTest.sh 
    === ./CSGQueryTest.sh catgeom ContiguousThreeSphere_XY override of default geom
    === ./CSGQueryTest.sh catgeom ContiguousThreeSphere_XY geom ContiguousThreeSphere GEOM ContiguousThreeSphere
    //intersect_prim typecode 11 
    //intersect_node_contiguous 
    //distance_leaf typecode 101 complement 0 sd    30.6563 
    //distance_node_list isub 0 sub_sd    30.6563 sd    30.6563 
    //distance_leaf typecode 101 complement 0 sd    30.6563 
    //distance_node_list isub 1 sub_sd    30.6563 sd    30.6563 
    //distance_leaf typecode 101 complement 0 sd  -100.0000 
    //distance_node_list isub 2 sub_sd  -100.0000 sd  -100.0000 
    //intersect_node_contiguous sd  -100.0000 inside_or_surface 1 num_sub 3 offset_sub 0 
    //intersect_leaf typecode 11 gtransformIdx 4 
    //intersect_leaf ray_origin (    0.0000,  -70.7107,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (    0.0000,  -70.7107,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 0  isect (     0.0000     0.0000     0.0000     0.0000)  
    //intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_leaf typecode 101 gtransformIdx 2 
    //intersect_leaf ray_origin (    0.0000,  -70.7107,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 0  isect (     0.0000     0.0000     0.0000     0.0000)  
    //intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_node_contiguous valid_intersect 0  (    0.0000     0.0000     0.0000     0.0000) 
                               One MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (    0.0000  -70.7107    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 







