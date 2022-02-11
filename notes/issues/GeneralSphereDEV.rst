GeneralSphereDEV
===================



Curious no isect from genstep IXYZ (0,0,0)
---------------------------------------------

::


    In [15]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 0) ]                                                                                
    Out[15]: array([], shape=(0, 4), dtype=int8)

    In [16]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == 1) ]                                                                                
    Out[16]: 
    array([[ 0,  0,  1,  3],
           [ 0,  0,  1,  5],
           [ 0,  0,  1, 11],
           [ 0,  0,  1, 14],
           [ 0,  0,  1, 20],
           [ 0,  0,  1, 33],
           [ 0,  0,  1, 43],
           [ 0,  0,  1, 52],
           [ 0,  0,  1, 53],
           [ 0,  0,  1, 54],
           [ 0,  0,  1, 57],
           [ 0,  0,  1, 87]], dtype=int8)

    In [17]: isect_gsid[ np.logical_and( np.logical_and( isect_gsid[:,0] == 0, isect_gsid[:,1] == 0), isect_gsid[:,2] == -1) ]                                                                               
    Out[17]: 
    array([[ 0,  0, -1,  4],
           [ 0,  0, -1, 16],
           [ 0,  0, -1, 32],
           [ 0,  0, -1, 46],
           [ 0,  0, -1, 54],
           [ 0,  0, -1, 56],
           [ 0,  0, -1, 57],
           [ 0,  0, -1, 77],
           [ 0,  0, -1, 86],
           [ 0,  0, -1, 99]], dtype=int8)


Using regular bicycle spoke photon directions with negative PHO 
----------------------------------------------------------------------


Unexpected miss in outer direction around XY plane::

     IXYZ=-5,0,0 PHO=-100 ./csg_geochain.sh 

Seems like photons with angle in the thetacut range failing to land. 

* need some special handling of unbounded 


CSG/csg_intersect_leaf.h::

    1485     if(complement)  // flip normal, even for miss need to signal the complement with a -0.f  
    1486     {
    1487         isect.x = -isect.x ;
    1488         isect.y = -isect.y ;
    1489         isect.z = -isect.z ;
    1490     }
    1491     /*
    1492 
    1493     // unbounded leaf cutters MISS need some special casing 
    1494     // unbounded MISS needs to be converted into EXIT but only in some directions 
    1495 
    1496     else if(valid_isect == false && typecode == CSG_THETACUT)
    1497     {
    1498         isect.x = -isect.x ;
    1499     }
    1500     */
    1501     
    1502     return valid_isect ;
    1503 }


The above is too indescriminate, need to do it only when the rays are headed for the 
otherside at infinity. 

TODO: try not constructing by intersecting but instead by intersecting with the complemented other side 




Using 2D(embedded in 3D) cross products : can determine if ray direction is between cone directions
---------------------------------------------------------------------------------------------------------



::

     IXYZ=9,0,9 ./csg_geochain.sh ana        # expected
     IXYZ=10,0,10 ./csg_geochain.sh ana      # no isect : duh : because there are no gensteps at that grid position



Debug blank cxr_geochain.sh render with thetacut : NOW FIXED 
----------------------------------------------------------------

Modify to a full sphere to orientate and prepare debug. 
Set EYE inside the sphere so every pixel should intersect::

    EYE=-0.25,0,0 ./cxr_geochain.sh 

Get logging::

    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3494 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3622 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3749 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2852 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2981 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3109 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.3238 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2333 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2463 
    //geo_OptiXTest.cu:intersect identity 0 primIdx 0 nodeOffset 0 numNode 1 valid_isect 1 isect.w   111.2593 


Now back to thetacut sphere and recreate the CSG geom::

   x4 ; vi GeneralSphereDEV.sh 
   gc ; ./run.sh 

Huh, working already. Must have been just that the was not seeing the new headers::

    EYE=-1,-1,1 TMIN=0.1 ./cxr_geochain.sh 



Onwards to phicut 
--------------------

Pacman, but failing to intersect with half of phi:: 

    IXYZ=-3,3,0 ./csg_geochain.sh 

Fixed this using::

    109 LEAF_FUNC
    110 bool intersect_leaf_phicut( float4& isect, const quad& q0, const float t_min, const float3& o, const float3& d )
    111 {
    112     const float& cosPhi0 = q0.f.x ;
    113     const float& sinPhi0 = q0.f.y ;
    114     const float& cosPhi1 = q0.f.z ;
    115     const float& sinPhi1 = q0.f.w ;
    116 
    117     const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    118     const float PR = cosPhi0*d.y - d.x*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    119     const float QR = cosPhi1*d.y - d.x*sinPhi1  ;
    120     bool unbounded_exit = PQ > 0.f ? ( PR > 0.f && QR < 0.f ) : ( PR > 0.f || QR < 0.f )  ;
    ...
    217     if( valid_intersect )
    218     {
    219         isect.x = t_cand == t1 ? -sinPhi1 :  sinPhi0 ;
    220         isect.y = t_cand == t1 ?  cosPhi1 : -cosPhi0 ;
    221         isect.z = 0.f ;
    222         isect.w = t_cand ;
    223     }
    224     else if( unbounded_exit )
    225     {
    226         isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity 
    227     }
    228 
    229     return valid_intersect ;
    230 }



Trying Lucas reduced resource imp has spurious intersects for rays starting on the phi0 line::

    SPHI=0.24,1.76 IXYZ=4,4,0 ./csg_geochain.sh ana


With "t_cand < t_min" get the spurious intersect:: 

    2022-02-11 18:52:23.170 INFO  [33207127] [CSGGeometry::saveCenterExtentGenstepIntersect@189] [ pp.size 62700 t_min     0.0000
    2022-02-11 18:52:23.171 INFO  [33207127] [CSGGeometry::saveCenterExtentGenstepIntersect@218] [ single photon selected
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (   60.0000,   60.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.3569,   -0.9341,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     0.9493    -0.3143     0.0000    97.8783)  
    //intersect_leaf valid_isect 1 isect (    0.9493    -0.3143     0.0000    97.8783)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.9493    -0.3143     0.0000   -97.8783) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin (   60.0000,   60.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.3569,   -0.9341,    0.0000) 
    //intersect_leaf_phicut q0.f  (    0.7071     0.7071     0.7071    -0.7071) cosPhi0/sinPhi0/cosPhi1/sinPhi1 
    //intersect_leaf_phicut d.xyz (     0.3569    -0.9341     0.0000 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 
    //intersect_leaf_phicut PR    -0.9129 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR    -0.9129 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 1 
    //intersect_leaf_phicut ( o.x*sinPhi0 + o.y*(-cosPhi0)        0.0000 
    //intersect_leaf_phicut ( d.x*sinPhi0 + d.y*(-cosPhi0)        0.9129 
    //intersect_leaf_phicut t_min        0.0000 
    //intersect_leaf_phicut t_cand.0    -0.0000 
    //intersect_leaf_phicut t_cand.1    -0.0000 
    //intersect_leaf_phicut t1        207.8780 
    //intersect_leaf_phicut t_cand.2    -0.0000 valid_intersect 0 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ THIS IS SPURIOUS INTERSECT -0.f ARISING FROM BUG : -0.f IS NOT GREATER THAN t_min = 0.f   

    //intersect_leaf valid_isect 0 isect (    0.0000    -0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.0000    -0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 
    2022-02-11 18:52:23.171 INFO  [33207127] [CSGGeometry::saveCenterExtentGenstepIntersect@221] 
    single photon selected INTERSECT
                        q0 norm t (    0.9493   -0.3143    0.0000   97.8783)
                       q1 ipos sd (   94.9314  -31.4328    0.0000   44.9003)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (   60.0000   60.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    0.3569   -0.9341    0.0000 C4U (     4    4    0   80 ) )

    2022-02-11 18:52:23.171 INFO  [33207127] [CSGGeometry::saveCenterExtentGenstepIntersect@222] ] single photon selected 
    2022-02-11 18:52:23.171 INFO  [33207127] [CSGGeometry::saveCenterExtentGenstepIntersect@227]  pp.size 62700 num_ray 1 ii.size 1 

Switching to "t_cand <= t_min" avoids the spurious intersect:: 

    2022-02-11 18:59:45.227 INFO  [33215201] [CSGGeometry::saveCenterExtentGenstepIntersect@189] [ pp.size 62700 t_min     0.0000
    2022-02-11 18:59:45.228 INFO  [33215201] [CSGGeometry::saveCenterExtentGenstepIntersect@218] [ single photon selected
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (   60.0000,   60.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.3569,   -0.9341,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     0.9493    -0.3143     0.0000    97.8783)  
    //intersect_leaf valid_isect 1 isect (    0.9493    -0.3143     0.0000    97.8783)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.9493    -0.3143     0.0000   -97.8783) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin (   60.0000,   60.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.3569,   -0.9341,    0.0000) 
    //intersect_leaf_phicut q0.f  (    0.7071     0.7071     0.7071    -0.7071) cosPhi0/sinPhi0/cosPhi1/sinPhi1 
    //intersect_leaf_phicut d.xyz (     0.3569    -0.9341     0.0000 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 
    //intersect_leaf_phicut PR    -0.9129 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR    -0.9129 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 1 
    //intersect_leaf_phicut ( o.x*sinPhi0 + o.y*(-cosPhi0)        0.0000 
    //intersect_leaf_phicut ( d.x*sinPhi0 + d.y*(-cosPhi0)        0.9129 
    //intersect_leaf_phicut t_min        0.0000 
    //intersect_leaf_phicut t_cand.0    -0.0000 
    //intersect_leaf_phicut t_cand.1 999999988484154753734934528.0000 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ^^^^^  
    ^^^^^     THIS IS THE DIFFERENCE BETWEEN  
    ^^^^^         WITH-BUG         if(signbit(o.x+t_cand*d.x) != signbit(cosPhi0) || t_cand <  t_min ) t_cand = RT_DEFAULT_MAX ;    
    ^^^^^         FIXED            if(signbit(o.x+t_cand*d.x) != signbit(cosPhi0) || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ;    
    ^^^^^  
    ^^^^^     t_cand.1 gets invalidated after the FIX,   before the FIX t_cand.1 becomes -0.f  
    ^^^^^  
    ^^^^^ 
    //intersect_leaf_phicut t1        207.8780 
    //intersect_leaf_phicut t_cand.2   207.8780 valid_intersect 1 
    //intersect_leaf valid_isect 1 isect (    0.7071     0.7071     0.0000   207.8780)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.7071     0.7071     0.0000   207.8780) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 
    //intersect_tree nodeIdx  1 height  1 depth  0 elevation  1 endTree    10000 leftTree  3020000 rightTree  1030000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (   60.0000,   60.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.3569,   -0.9341,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 0  isect (     0.0000     0.0000     0.0000     0.0000)  
    //intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.0000     0.0000     0.0000    -0.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 
    2022-02-11 18:59:45.228 INFO  [33215201] [CSGGeometry::saveCenterExtentGenstepIntersect@221] 
    single photon selected no intersect
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (   60.0000   60.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    0.3569   -0.9341    0.0000 C4U (     4    4    0   80 ) )

    2022-02-11 18:59:45.228 INFO  [33215201] [CSGGeometry::saveCenterExtentGenstepIntersect@222] ] single photon selected 
        



