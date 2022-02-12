GeneralSphereDEV
===================


Curious no isect from genstep IXYZ (0,0,0)  : FIXED using unbounded_exit : isect.y -0.f to signal changing a MISS to an EXIT at infinity 
-------------------------------------------------------------------------------------------------------------------------------------------

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
--------------------------------------------------------------------------


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
        



3D renders of GeneralSphereDEV with pacman phicut :  Spurious line of intersects along the axis line
-------------------------------------------------------------------------------------------------------

::

    EYE=1,0,1 TMIN=0. ./cxr_geochain.sh 
        clear line "tail" extending along axis beneath S pole 

        * after change to judging wrong side based on y (rather than x) no longer see the tail

    EYE=1,0,0.5 TMIN=0 ./cxr_geochain.sh 
        still there but less obvious 

        * with y wrong side judging, no artifacts visible


    EYE=1,0,-1 TMIN=0.1 ./cxr_geochain.sh 
        from benath see the line extending along axis above N pole  

        * artifacts gone here too when use y to judge phicut side


    EYE=-1,0,-1 TMIN=0.1 ./cxr_geochain.sh 
        from backside dont see any tail : but do see a few white dot misses along axis : like a perforation   

        * same again : no visible artifacts when use y 

    EYE=1,1,1 TMIN=0.1 ./cxr_geochain.sh 
         viewing from a point within the phi0 plane not causing artifacts   

         * still OK using y

    EYE=-0.99,-1,1 TMIN=0.1 ./cxr_geochain.sh 
         visible vertical axis seam of misses between hemisphere to left and quadrant to the right 

         * using Y : the seam is still visible

    EYE=-1,-1,1 TMIN=0.1 ./cxr_geochain.sh 
         huh : only see the quadrant to the right, the hemi to the left disappears

         * using Y : still same with missing left hemi
         * problem is with rays travelling in (1,1,-1) direction : which is within one of the phicut planes
         * is it just the direction or also due to origin being within the plane ?

           * it must be just the direction causing the problem otherwise would never get the entire hemi to disappear 

    EYE=-1,-1,1 TMIN=0.1 LOOK=0.001,0,0  ./cxr_geochain.sh 
         moving the look point regains the geometry 

    EYE=-1,-1,1 TMIN=0.1 CAM=0  ./cxr_geochain.sh 
         with perspective camera nothing missing : but clear axis seam 

    EYE=-1,-1,0 TMIN=0.1 CAM=0 ./cxr_geochain.sh
         with perspective camera, nothing missing because the rays are fanning out : but clear axis seam 

    EYE=-1,-1,-1 TMIN=0.1 ./cxr_geochain.sh 
         similar : hemi to the left has disappeared

         * same using Y

    EYE=-1.001,-1.001,-1.001 TMIN=0.1 ./cxr_geochain.sh 
         same again : hemi to left missing 

    EYE=-1.001,-1,1 TMIN=0.1 ./cxr_geochain.sh 
         expected render back again, no seam or anything missing 



    GEOM=GeneralSphereDEV_XYZ NO_GS=1 EDL=1 ./csg_geochain.sh 
         no spurious with 3D gensteps : get issues when precisely lined up with things
         in 3D it is unlikely to be precisely lined up with anything 
 
    
 


TODO: try to reproduce these 2 forms of misbehavuior in 2D  XZ projection 

::

     SPURIOUS=1 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana 

     IXYZ=1,0,0 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana 

     IXYZ=1,0,0 SPURIOUS=1 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana 

     SPURIOUS=1 PLOT_SELECTED=1 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana 
         getting righthand side arc which should be cut away 

     SPURIOUS=1 PLOT_SELECTED=1 IXYZ=10,0,0 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana 
          4 splash back spurious

     IXYZ=10,0,0 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh ana
          clearly shows 4 spurious, with most hitting the flat edge as expected





IXYZ=10,0,0 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh::


     .                             count_all : 18446 
                              count_spurious : 1706 
                            count_isect_gsid : 18   IXYZ 10,0,0  ix:10 iy:0 iz:0
                                count_select : 18  

                                count_select : 18  

                                     s_count : 18  

                                   s_limited : 18  

                              selected_isect : 18  

      s_isect_gsid (  10   0   0  41 )   s_t (   174.8285 )   s_pos (     0.0000     0.0000    89.8055 )   s_pos_r (    89.8055 )   s_sd (     0.0000 )  

      s_isect_gsid (  10   0   0  42 )   s_t (    60.6917 )   s_pos (    96.0551     0.0000    27.8105 )   s_pos_r (   100.0000 )   s_sd (    67.9212 )  

      s_isect_gsid (  10   0   0  43 )   s_t (   163.7361 )   s_pos (     0.0000     0.0000    65.6468 )   s_pos_r (    65.6468 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  44 )   s_t (   159.6266 )   s_pos (     0.0000     0.0000    54.5955 )   s_pos_r (    54.5955 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  45 )   s_t (   156.3326 )   s_pos (     0.0000     0.0000    44.0440 )   s_pos_r (    44.0440 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  46 )   s_t (   153.7784 )   s_pos (     0.0000     0.0000    33.8790 )   s_pos_r (    33.8790 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  47 )   s_t (    50.9649 )   s_pos (    99.6753     0.0000     8.0525 )   s_pos_r (   100.0000 )   s_sd (    70.4810 )  
      s_isect_gsid (  10   0   0  48 )   s_t (   150.6823 )   s_pos (     0.0000     0.0000    14.3232 )   s_pos_r (    14.3232 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  49 )   s_t (   150.0756 )   s_pos (     0.0000     0.0000     4.7616 )   s_pos_r (     4.7616 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  50 )   s_t (   150.0756 )   s_pos (     0.0000     0.0000    -4.7616 )   s_pos_r (     4.7616 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  51 )   s_t (   150.6823 )   s_pos (     0.0000     0.0000   -14.3233 )   s_pos_r (    14.3233 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  52 )   s_t (   151.9081 )   s_pos (     0.0000     0.0000   -24.0017 )   s_pos_r (    24.0017 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  53 )   s_t (    51.9327 )   s_pos (    99.3433     0.0000   -11.4413 )   s_pos_r (   100.0000 )   s_sd (    70.2463 )  
      s_isect_gsid (  10   0   0  54 )   s_t (   156.3326 )   s_pos (     0.0000     0.0000   -44.0440 )   s_pos_r (    44.0440 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  55 )   s_t (   159.6266 )   s_pos (     0.0000     0.0000   -54.5955 )   s_pos_r (    54.5955 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  56 )   s_t (   163.7361 )   s_pos (     0.0000     0.0000   -65.6468 )   s_pos_r (    65.6468 )   s_sd (     0.0000 )  
      s_isect_gsid (  10   0   0  57 )   s_t (    60.6917 )   s_pos (    96.0551     0.0000   -27.8106 )   s_pos_r (   100.0000 )   s_sd (    67.9212 )  
      s_isect_gsid (  10   0   0  58 )   s_t (   174.8286 )   s_pos (     0.0000     0.0000   -89.8055 )   s_pos_r (    89.8055 )   s_sd (     0.0000 )  


These are intersects onto the open edge of pacmans mouth. Imprecision making the side dip into the wrong sign will cause spurious. 

Rerunning the first spurious (  10   0   0  42 ) reproduces it::

    epsilon:CSG blyth$ SXYZW=10,0,0,42 GEOM=GeneralSphereDEV_XZ ./csg_geochain.sh run
    ...
    2022-02-12 15:02:42.431 INFO  [33772776] [CSGGeometry::init_fd@91]  fd.meta
    creator:X4SolidMaker::GeneralSphereDEV
    name:GeneralSphereDEV
    innerRadius:0
    outerRadius:100
    phiStart:0.25
    phiDelta:1.5
    thetaStart:0
    thetaDelta:1
    ...
    2022-02-12 15:02:42.431 INFO  [33772776] [CSGGeometry::init_selection@134] SXYZW (sx,sy,sz,sw) (10,0,0,42)
    2022-02-12 15:02:42.453 INFO  [33772776] [CSGGeometry::saveCenterExtentGenstepIntersect@189] [ pp.size 62700 t_min     0.0000
    2022-02-12 15:02:42.454 INFO  [33772776] [CSGGeometry::saveCenterExtentGenstepIntersect@218] [ single photon selected
    2022-02-12 15:02:42.454 INFO  [33772776] [CSGGeometry::saveCenterExtentGenstepIntersect@221] 
            single photon selected HIT
                        q0 norm t (    0.9606    0.0000    0.2781   60.6917)
                       q1 ipos sd (   96.0551    0.0000   27.8105   67.9212)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  150.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (   -0.8888    0.0000    0.4582 C4U (    10    0    0   42 ) )

    2022-02-12 15:02:42.454 INFO  [33772776] [CSGGeometry::saveCenterExtentGenstepIntersect@222] ] single photon selected 
    2022-02-12 15:02:42.587 INFO  [33772776] [CSGDraw::draw@27] CSGGeometry::centerExtentGenstepIntersect axis Y

               in                           
              1                             
                 0.00                       
                -0.00                       
                                            
     sp                  ph                 
    2                   3                   
     100.00              100.00             
    -100.00             -100.00             
                                            
                                            
                                           
So recompile with DEBUG flag and repeat to see whats happening.
The intersect_x onto the cut face is slightly negative causing it to be classified 
as wrong side::
     
    //intersect_leaf ray_direction (   -0.8888,    0.0000,    0.4582) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     0.9606     0.0000     0.2781    60.6917)  
    //intersect_leaf valid_isect 1 isect (    0.9606     0.0000     0.2781    60.6917)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.9606     0.0000     0.2781   -60.6917) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin (  150.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (   -0.8888,    0.0000,    0.4582) 
    //intersect_leaf_phicut q0.f  (    0.7071     0.7071     0.7071    -0.7071) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     0.0000 
    //intersect_leaf_phicut d.xyz (    -0.8888     0.0000     0.4582 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve angle less than pi, -ve angle greater than pi 
    //intersect_leaf_phicut PR     0.6285 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR     0.6285 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 1 
    //intersect_leaf_phicut t_cand.0   168.7601 t_min     0.0000 
    //intersect_leaf_phicut intersect_x    -0.0000 intersect_x*1e6   -15.2588 cosPhi0     0.7071 wrong_side 1 too_close 0 invalidate_t_cand_0 1 
    //intersect_leaf_phicut t_cand.1 999999988484154753734934528.0000 
    //intersect_leaf_phicut t_cand.2 999999988484154753734934528.0000 valid_intersect 0 
    //intersect_leaf valid_isect 0 isect (    0.0000    -0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.0000    -0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
                                            
                                       

Offsetting by 1e-4 avoids the spurious from this direction.::

    if((o.x+t_cand*d.x+1e-4)*cosPhi0 < 0.f || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ;

Can the y help ?  It is also real close to zero along the axis

Curious swiching to y avoids the problem in XZ projection testing::

    228     if((o.y+t_cand*d.y)*sinPhi0 < 0.f || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ;
    229     //if((o.x+t_cand*d.x+1e-4)*cosPhi0 < 0.f || t_cand <= t_min ) t_cand = RT_DEFAULT_MAX ; 
    230     //if((o.x+t_cand*d.x)*cosPhi0 < 0.f || t_cand < t_min ) t_cand = RT_DEFAULT_MAX ;         // t_cand < t_min YIELDS SPURIOUS INTERSECTS 
    231 /*





The Above investigations are all will pacman : now modify to pacmanpp aligning the phi plane with the axes : for easier tickling of the issue
-------------------------------------------------------------------------------------------------------------------------------------------------

x4 ; GeneralSphereDEV.sh::

     67 #phiMode=full
     68 #phiMode=melon
     69 #phiMode=pacman
     70 phiMode=pacmanpp
     71 
     72 case $phiMode in
     73         full)    phiStart=0.00 ; phiDelta=2.00 ;;
     74        melon)    phiStart=0.25 ; phiDelta=0.50 ;;
     75       pacman)    phiStart=0.25 ; phiDelta=1.50 ;;
     76       pacmanpp)  phiStart=0.50 ; phiDelta=1.50 ;;
     77 esac


Geant4 giving spurious intersects::

   x4
   GEOM=GeneralSphereDEV_XY ./xxs.sh 


Curious dont see spurious in any projection with CSG, hmm maybe aligning the planes with the axes makes the math on less of a precision knife-edge::

   c
   csg_geochain.sh 


3D renders::

    EYE=-2,0,0 TMIN=0.1 CAM=1 ./cxr_geochain.sh 
        parallel projection, very clear white seam line along axis 

        rays travelling in +X direction (1,0,0) from positions like (-100,0,-100->100) all miss the geometry 
        these rays are in one of the phicut planes (expect phi1)  

    EYE=2,0,0 TMIN=0.1 CAM=1 ./cxr_geochain.sh 
        looking back the other way, no artifact visible

    EYE=-1,0,0 TMIN=0. CAM=1 ./cxr_geochain.sh 
        again clear white seam, this position slightly cuts into the sphere  

    EYE=-1.1,0,0 TMIN=0. CAM=1 ./cxr_geochain.sh 
        bizarre no seam from -1.1 -1.2 -1.3 but clear seam at -1.5

        * so its a numerical precision handling an edge issue 

    EYE=-1.5,0,0 TMIN=0. CAM=1 ./cxr_geochain.sh 
        clear seam : back view 

    EYE=-1.5,0,0 TMIN=0. CAM=1 ZOOM=5 ./cxr_geochain.sh
        zooming in so the entire frame has intersects, except for the white seam down the middle

    EYE=-1.5,0,0 TMIN=0. CAM=0 ./cxr_geochain.sh 
        even perspective cam shows seam line  
    


Add an isect_buffer to OptiX6Test.cu : that suggests that all pixels are hits. 

Add cx tests/CSGOptiXRenderTest.py to imshow the result from isect_buffer 
shows that the "white" seam is actually not a line of misses. It is  
a line of different lighter coloring. So that suggests the problem is a line 
of somehow different normals.  

Looking at the 3x3 central pixel isects::

    epsilon:CSGOptiX blyth$ i tests/CSGOptiXRenderTest.py 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) :  select the central portion of the image array  
    ...    

    In [12]: b[:,0]                                                                                                                                                                                      
    Out[12]: 
    array([[[   0.   ,    0.5  ,    0.5  ,    0.   ],      prd.result
            [-100.   ,    0.056,    0.   ,    0.   ],      prd.posi        : thats expected for radius 100
            [-150.   ,    0.056,    0.   ,    1.5  ],      origin, tmin
            [   1.   ,    0.   ,    0.   ,    0.   ]],     direction, mode(int) 

           [[   0.   ,    0.5  ,    0.5  ,    0.   ],
            [-100.   ,    0.056,   -0.056,    0.   ],
            [-150.   ,    0.056,   -0.056,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.5  ,    0.499,    0.   ],
            [-100.   ,    0.056,   -0.111,    0.   ],
            [-150.   ,    0.056,   -0.111,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [13]: b[:,2]                                                                                                                                                                                      
    Out[13]: 
    array([[[   0.   ,    0.5  ,    0.5  ,    0.   ],
            [-100.   ,   -0.056,    0.   ,    0.   ],
            [-150.   ,   -0.056,    0.   ,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.5  ,    0.5  ,    0.   ],
            [-100.   ,   -0.056,   -0.056,    0.   ],
            [-150.   ,   -0.056,   -0.056,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.5  ,    0.499,    0.   ],
            [-100.   ,   -0.056,   -0.111,    0.   ],
            [-150.   ,   -0.056,   -0.111,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)


    In [14]: b[:,1]                                                                                                                                                                                      
    Out[14]: 
    array([[[   0.5  ,    1.   ,    0.5  ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],     prd.posi   NOT EXPECTED (THIS IS A BACK VIEW) : GETTING HIT AT ORIGIN  
            [-150.   ,    0.   ,    0.   ,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.5  ,    1.   ,    0.5  ,    0.   ],
            [   0.   ,    0.   ,   -0.056,    0.   ],     prd.posi    ALSO HITS UP THE Z AXIS LINE
            [-150.   ,    0.   ,   -0.056,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.5  ,    1.   ,    0.5  ,    0.   ],
            [   0.   ,    0.   ,   -0.111,    0.   ],
            [-150.   ,    0.   ,   -0.111,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)



So now I know the ray origin and direction for the unexpected pixels::

    DUMP=3 ORI=-150,0,0 DIR=1,0,0 CSGQueryTest O


    epsilon:CSG blyth$ DUMP=3 ORI=-150,0,0 DIR=1,0,0 CSGQueryTest O

    2022-02-12 22:18:56.223 INFO  [34231258] [CSGFoundry::load@1345] /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGFoundry
    ...
    2022-02-12 22:18:56.225 INFO  [34231258] [*CSGFoundry::LoadGeom@1414] creator:X4SolidMaker::GeneralSphereDEV
    name:GeneralSphereDEV
    innerRadius:0
    outerRadius:100
    phiStart:0.5
    phiDelta:1.5
    thetaStart:0
    thetaDelta:1
    2022-02-12 22:18:56.225 INFO  [34231258] [CSGQuery::selectPrim@60]  select_prim 0x108ca8000 select_nodeOffset 0 select_numNode 3 select_root 0x108fb6000 select_root_typecode intersection getSelectedTreeHeight 1
    2022-02-12 22:18:56.225 INFO  [34231258] [CSGQueryTest::CSGQueryTest@77]  GEOM GeneralSphereDEV
    2022-02-12 22:18:56.225 INFO  [34231258] [CSGDraw::draw@27] CSGQueryTest axis Y

               in                           
              1                             
                 0.00                       
                -0.00                       
                                            
     sp                  ph                 
    2                   3                   
     100.00              100.00             
    -100.00             -100.00             
                                            
                                            
                                            
                                            
                                            
                                            
    2022-02-12 22:18:56.225 INFO  [34231258] [CSGQueryTest::operator@83]  mode O
    2022-02-12 22:18:56.225 INFO  [34231258] [CSGQueryTest::config@134] 
     name One
     dump 3 dump_hit 1 dump_miss 1 ( 0:no 1:hit 2:miss 3:hit+miss ) 
     ORI ray_origin (-150.000, 0.000, 0.000) 
     DIR ray_direction ( 1.000, 0.000, 0.000) 
     TMIN tmin 0.000
     GSID gsid 0
     NUM num 1

                               One HIT
                        q0 norm t (   -0.0000    1.0000    0.0000  150.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -150.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 
    epsilon:CSG blyth$ 



Recompile with DEBUG flag and rerun that ray, why is the further hit trumping the nearer one::

    2022-02-12 22:22:32.764 INFO  [34236727] [CSGQueryTest::operator@83]  mode O
    2022-02-12 22:22:32.764 INFO  [34236727] [CSGQueryTest::config@134] 
     name One
     dump 3 dump_hit 1 dump_miss 1 ( 0:no 1:hit 2:miss 3:hit+miss ) 
     ORI ray_origin (-150.000, 0.000, 0.000) 
     DIR ray_direction ( 1.000, 0.000, 0.000) 
     TMIN tmin 0.000
     GSID gsid 0
     NUM num 1

    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -150.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (    -1.0000     0.0000     0.0000    50.0000)  
    //intersect_leaf valid_isect 1 isect (   -1.0000     0.0000     0.0000    50.0000)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (   -1.0000     0.0000     0.0000   -50.0000) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin ( -150.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_phicut q0.f  (   -0.0000     1.0000     1.0000     0.0000) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     0.0000 
    //intersect_leaf_phicut d.xyz (     1.0000     0.0000     0.0000 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve angle less than pi, -ve angle greater than pi 
    //intersect_leaf_phicut PR    -1.0000 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR    -1.0000 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 1 
    //intersect_leaf_phicut t_cand.0   150.0000 t_min     0.0000 
    //intersect_leaf_phicut ipos_x     0.0000 ipos_x*1e6f     0.0000  cosPhi0    -0.0000  x_wrong_side 0 
    //intersect_leaf_phicut ipos_y     0.0000 ipos_y*1e6f     0.0000  sinPhi0     1.0000  y_wrong_side 0 
    //intersect_leaf_phicut t_cand   150.0000 t_min     0.0000 too_close 0 
    //intersect_leaf_phicut t_cand.1   150.0000 
    //intersect_leaf_phicut t_cand.2   150.0000 valid_intersect 1 
    //intersect_leaf valid_isect 1 isect (   -0.0000     1.0000     0.0000   150.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (   -0.0000     1.0000     0.0000   150.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 
    //intersect_tree nodeIdx  1 height  1 depth  0 elevation  1 endTree    10000 leftTree  3020000 rightTree  1030000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -150.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    1.0000,    0.0000,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     1.0000     0.0000     0.0000   250.0000)  
    //intersect_leaf valid_isect 1 isect (    1.0000     0.0000     0.0000   250.0000)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    1.0000     0.0000     0.0000  -250.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 
                               One HIT
                        q0 norm t (   -0.0000    1.0000    0.0000  150.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -150.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 


Artificially setting unbounded_exit to always false does not change the outcome. 

