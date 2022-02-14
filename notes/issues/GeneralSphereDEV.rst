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
         * from future : note that the seam is not from misses, it is from unexpected hits onto the axis line 

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

        * after SMath::sin_pi/cos_pi : right side hemi disappears
        * after unbounded exit permit 0. : right side hemi is back again 

    EYE=-1.5,0,0 TMIN=0. CAM=0 ./cxr_geochain.sh


    EYE=-1.5,0,0 TMIN=0. CAM=1 ZOOM=5 ./cxr_geochain.sh
        zooming in so the entire frame has intersects, except for the white seam down the middle

        * after SMath::sin_pi/cos_pi : entire RHS is miss 



    EYE=-1.5,0,0 TMIN=0. CAM=0 ./cxr_geochain.sh 
        even perspective cam shows seam line  
    
        * after SMath::sin_pi/cos_pi : nothing missing, no seam 


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

    epsilon:CSG blyth$ DUMP=3 ORI=-150,0,0 DIR=1,0,0 CSGQueryTest O       ## ./CSGQueryTest.sh 

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

Notice the normal direction for intersects onto the axis line. 
They are perpendicular to the ray direction which will mess up hit classification
which is based on the dot product of ray direction and the normal. 
When that is zero it is possible that an EXIT is mis-classified as an ENTER.


::

        .                        
                                      Y
                                  .   -     
                              .       |         
                                      |    pp quadrant cutaway with pacmanpp
                           .          |           
                                      ^  (0,1,0) normal
                          .          /|\            
                                      |             
        -X  0 -----------1------------2--->---------3------------------- +X
                                      |  (1,0,0) dir
         -150          -100           0            100
                                      |  
                           .          |          .
                                      |      
                              .       |       .
                                  .   -   .




More debug, confirms the cause is an ENTER that should be an EXIT::

    ./CSGQueryTest.sh 

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

    //intersect_tree nodeIdx   1 t_left    50.0000 t_right   150.0000 leftIsCloser 1  l_state Enter r_state Enter l_cos*1e6f -1000000.0000 r_cos*1e6f    -0.1748 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter    50.0000    r:Enter   150.0000     leftIsCloser 1 -> LOOP_A 
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

    //intersect_tree nodeIdx   1 t_left   250.0000 t_right   150.0000 leftIsCloser 0  l_state  Exit r_state Enter l_cos*1e6f 1000000.0000 r_cos*1e6f    -0.1748 
    //   1 : stack peeking : left 1 right 0 (stackIdx)     intersection  l: Exit   250.0000    r:Enter   150.0000     leftIsCloser 0 -> RETURN_B 
                               One HIT
                        q0 norm t (   -0.0000    1.0000    0.0000  150.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -150.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )


Classification comes from dot product::

    #define CSG_CLASSIFY( ise, dir, tmin )   (fabsf((ise).w) > (tmin) ?  ( (ise).x*(dir).x + (ise).y*(dir).y + (ise).z*(dir).z < 0.f ? State_Enter : State_Exit ) : State_Miss )

    < 0.f  : ENTER   direction is against the normal 
    >= 0.f : EXIT    direction is with the normal 

So next question is where is the very slightly -ve x component of the normal coming from.::

    278     if( valid_intersect )
    279     {
    280         isect.x = t_cand == t1 ? -sinPhi1 :  sinPhi0 ;
    281         isect.y = t_cand == t1 ?  cosPhi1 : -cosPhi0 ;
    282         isect.z = 0.f ;
    283         isect.w = t_cand ;
    284     }
    285     else if( unbounded_exit )
    286     {
    287         isect.y = -isect.y ;  // -0.f signflip signalling that can promote MISS to EXIT at infinity 
    288     }
    289 



Its coming from -sinPhi1 = -sinPhi(2*pi) not being zero as it should be but a very small -ve value.
Changing that with SPhiCut and SMath::sin_pi and then rerun geochain and the intersect::


    2022-02-13 14:34:57.743 INFO  [34690282] [CSGQueryTest::config@134] 
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
    //intersect_leaf_phicut q0.f  (    0.0000     1.0000     1.0000     0.0000) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     0.0000 
    //intersect_leaf_phicut d.xyz (     1.0000     0.0000     0.0000 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve angle less than pi, -ve angle greater than pi 
    //intersect_leaf_phicut PR    -1.0000 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR    -1.0000 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 0 
    //intersect_leaf_phicut t_cand.0   150.0000 t_min     0.0000 
    //intersect_leaf_phicut ipos_x     0.0000 ipos_x*1e6f     0.0000  cosPhi0     0.0000  x_wrong_side 0 
    //intersect_leaf_phicut ipos_y     0.0000 ipos_y*1e6f     0.0000  sinPhi0     1.0000  y_wrong_side 0 
    //intersect_leaf_phicut t_cand   150.0000 t_min     0.0000 too_close 0 
    //intersect_leaf_phicut t_cand.1   150.0000 
    //intersect_leaf_phicut t_cand.2   150.0000 valid_intersect 1 
    //intersect_leaf valid_isect 1 isect (    1.0000     0.0000     0.0000   150.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    1.0000     0.0000     0.0000   150.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left    50.0000 t_right   150.0000 leftIsCloser 1  l_state Enter r_state  Exit l_cos*1e6f -1000000.0000 r_cos*1e6f 1000000.0000 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter    50.0000    r: Exit   150.0000     leftIsCloser 1 -> RETURN_A 
                               One HIT
                        q0 norm t (   -1.0000    0.0000    0.0000   50.0000)
                       q1 ipos sd ( -100.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -150.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 







CSGClassifyTest I 


+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| intersection    | B Enter         | B Exit          | B Miss          |                 
| A Closer        |                 |                 |                 |                 
| B Closer        |                 |                 |                 |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Enter         |                 |                 |                 |                 
|                 | LOOP_A          | RETURN_A        | RETURN_MISS     |                 
|                 | LOOP_B          | LOOP_B          | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Exit          |                 |                 |                 |                 
|                 | LOOP_A          | RETURN_A        | RETURN_MISS     |                 
|                 | RETURN_B        | RETURN_B        | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 
|                 |                 |                 |                 |                 
| A Miss          |                 |                 |                 |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 | RETURN_MISS     | RETURN_MISS     | RETURN_MISS     |                 
|                 |                 |                 |                 |                 
+-----------------+-----------------+-----------------+-----------------+                 





After switching to more precise SMath::sin_pi check the seam again::

    cx
    DXDY=5,5 i tests/CSGOptiXRenderTest.py 


Observe that its no longer a solid pale green line, but now more dotted and whiter line.
So now are actually getting misses. 

Back to old X -150 position, where get missing RHS::

    epsilon:CSGOptiX blyth$ DXDY=1,1 i tests/CSGOptiXRenderTest.py 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) :  select the central portion of the image array  


    In [1]: b                                                                                                                                                                                            
    In [2]: b.view(np.int32)[:,:,3,3]                                                                                                                                                                    
    Out[2]: 
    array([[2, 2, 1],               # mode:1 MISS  mode:2 HIT 
           [2, 2, 1],
           [2, 2, 1]], dtype=int32)


Right hand column of misses::

    In [4]: b[:,2]                                                                                                                                                                                       
    Out[4]: 
    array([[[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-150.   ,   -0.056,    0.   ,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-150.   ,   -0.056,   -0.056,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-150.   ,   -0.056,   -0.111,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)


Left hand column of hits onto the sphere, close to (-100,0,0)::

    In [5]: b[:,0]                                                                                                                                                                                       
    Out[5]: 
    array([[[   0.   ,    0.5  ,    0.5  ,    0.   ],
            [-100.   ,    0.056,    0.   ,    0.   ],
            [-150.   ,    0.056,    0.   ,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.5  ,    0.5  ,    0.   ],
            [-100.   ,    0.056,   -0.056,    0.   ],
            [-150.   ,    0.056,   -0.056,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.5  ,    0.499,    0.   ],
            [-100.   ,    0.056,   -0.111,    0.   ],
            [-150.   ,    0.056,   -0.111,    1.5  ],
            [   1.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)




Reproduce the miss nearby::

    epsilon:CSG blyth$ DUMP=3 ORI=-150,-1,-1 DIR=1,0,0 CSGQueryTest O
    2022-02-13 15:26:15.916 INFO  [34747756] [CSGFoundry::load@1345] /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGFoundry
    2022-02-13 15:26:15.918 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     1 nj 3 nk 4 solid.npy
    2022-02-13 15:26:15.918 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     1 nj 4 nk 4 prim.npy
    2022-02-13 15:26:15.918 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     3 nj 4 nk 4 node.npy
    2022-02-13 15:26:15.918 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     2 nj 4 nk 4 tran.npy
    2022-02-13 15:26:15.918 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     2 nj 4 nk 4 itra.npy
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGFoundry::loadArray@1476]  ni     1 nj 4 nk 4 inst.npy
    NP::load Failed to load from path /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGFoundry/bnd.npy
    NP::load Failed to load from path /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGFoundry/icdf.npy
    2022-02-13 15:26:15.919 INFO  [34747756] [*CSGFoundry::LoadGeom@1414] creator:X4SolidMaker::GeneralSphereDEV
    name:GeneralSphereDEV
    innerRadius:0
    outerRadius:100
    phiStart:0.5
    phiDelta:1.5
    thetaStart:0
    thetaDelta:1
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGQuery::selectPrim@60]  select_prim 0x10bb78000 select_nodeOffset 0 select_numNode 3 select_root 0x10be86000 select_root_typecode intersection getSelectedTreeHeight 1
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGQueryTest::CSGQueryTest@77]  GEOM GeneralSphereDEV
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGDraw::draw@27] CSGQueryTest axis Y

               in                           
              1                             
                 0.00                       
                -0.00                       
                                            
     sp                  ph                 
    2                   3                   
     100.00              100.00             
    -100.00             -100.00             
                                            
                                            
                                            
                                            
                                            
                                            
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGQueryTest::operator@83]  mode O
    2022-02-13 15:26:15.919 INFO  [34747756] [CSGQueryTest::config@134] 
     name One
     dump 3 dump_hit 1 dump_miss 1 ( 0:no 1:hit 2:miss 3:hit+miss ) 
     ORI ray_origin (-150.000,-1.000,-1.000) 
     DIR ray_direction ( 1.000, 0.000, 0.000) 
     TMIN tmin 0.000
     GSID gsid 0
     NUM num 1

                               One MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -150.0000   -1.0000   -1.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 



   
                                               Y
                                              (0,1) 
                                        . (cosPhi0, sinPhi0)
                                               |
                                    .          | 
                                               |
                                .              | 
                                               | 
                              .                |
                                               |
                 +-----------|-----------------0----------------------(1,0)--- X
                                                                    (cosPhi1, sinPhi1)
                              .
                 0 - - - - - - 1 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                              ENTER                                                      EXIT_PHICUT_AT_INFINITY
                              SPHERE 



::

    117     // PQ, PR, QR are 2D cross products used to identity unbounded exits at infinity 
    118     // see cross2D_angle_range_without_trig.py for explanation
    119     const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    120     const float PR = cosPhi0*d.y - d.x*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    121     const float QR = cosPhi1*d.y - d.x*sinPhi1  ;
    122     bool unbounded_exit = PQ > 0.f ? ( PR > 0.f && QR < 0.f ) : ( PR > 0.f || QR < 0.f )  ;
    123     //bool unbounded_exit = false ; 


Change to permitting 0.::

    117     // PQ, PR, QR are 2D cross products used to identity unbounded exits at infinity 
    118     // see cross2D_angle_range_without_trig.py for explanation
    119     const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    120     const float PR = cosPhi0*d.y - d.x*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    121     const float QR = cosPhi1*d.y - d.x*sinPhi1  ;
    122     bool unbounded_exit = PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f )  ;
    123     //bool unbounded_exit = false ; 


Note that have to manually touch cx .cu in order to pick up changed CSG headers in cx build


::

    EYE=-1.5,0,0  TMIN=0. CAM=1 ./cxr_geochain.sh 
        as expected from -X : cutaway not visible 
        after unbounded exit permitting zero 

        * RHS (-Y) hemi no longer missing, no seam 


    EYE=1.5,0,0  TMIN=0. CAM=1 ./cxr_geochain.sh 
        as expected from +X : cutaway +Y quadrant on the right with flat phicut face  

    EYE=0,0,2 UP=0,1,0  TMIN=0. CAM=0 ./cxr_geochain.sh 
        from +Z with perspective camera get expected cutaway pp-quadrant to top right 

    EYE=0,0,2 UP=0,1,0  TMIN=0. CAM=1 ./cxr_geochain.sh 
        from +Z with parallel camera get unexpected : full sphere 
        
    DUMP=3 ORI=1,1,200 DIR=0,0,-1 CSGQueryTest O  
        this is giving HIT when miss expected 


phicut is missed, but there is improper setting of unbounded exit which promotes the miss to EXIT::

    epsilon:CSG blyth$ DUMP=3 ORI=1,1,200 DIR=0,0,-1  CSGQueryTest O
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin (    1.0000,    1.0000,  200.0000) 
    //intersect_leaf ray_direction (    0.0000,    0.0000,   -1.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     0.0100     0.0100     0.9999   100.0100)  
    //intersect_leaf valid_isect 1 isect (    0.0100     0.0100     0.9999   100.0100)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.0100     0.0100     0.9999  -100.0100) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin (    1.0000,    1.0000,  200.0000) 
    //intersect_leaf ray_direction (    0.0000,    0.0000,   -1.0000) 
    //intersect_leaf_phicut q0.f  (    0.0000     1.0000     1.0000     0.0000) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     0.0000 
    //intersect_leaf_phicut d.xyz (     0.0000     0.0000    -1.0000 ) 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve:angle less than pi, -ve:angle greater than pi 
    //intersect_leaf_phicut PR     0.0000 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR     0.0000 cosPhi1*d.y - d.x*sinPhi1 

    A-HA : FOR PURE Z-DIRECTION-RAYS : THIS APPROACH CANNOT WORK AS PR AND QR ALWAYS ZERO   


    //intersect_leaf_phicut unbounded_exit 1 
    //intersect_leaf_phicut t_cand.0       -inf t_min     0.0000 
    //intersect_leaf_phicut ipos_x        nan ipos_x*1e6f        nan  cosPhi0     0.0000  x_wrong_side 0 
    //intersect_leaf_phicut ipos_y        nan ipos_y*1e6f        nan  sinPhi0     1.0000  y_wrong_side 0 
    //intersect_leaf_phicut t_cand       -inf t_min     0.0000 too_close 1 
    //intersect_leaf_phicut t_cand.1 999999988484154753734934528.0000 
    //intersect_leaf_phicut t_cand.2 999999988484154753734934528.0000 valid_intersect 0 
    //intersect_leaf valid_isect 0 isect (    0.0000    -0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.0000    -0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left   100.0100 t_right     0.0000 leftIsCloser 0  l_state Enter r_state  Miss l_cos*1e6f -999900.0000 r_cos*1e6f     0.0000 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter   100.0100    r: Exit     0.0000     leftIsCloser 1 -> RETURN_A 
                               One HIT
                        q0 norm t (    0.0100    0.0100    0.9999  100.0100)
                       q1 ipos sd (    1.0000    1.0000   99.9900    1.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (    1.0000    1.0000  200.0000    0.0000)
                  q3 ray_dir gsid (    0.0000    0.0000   -1.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 




Add zpure special case::

    144 LEAF_FUNC
    145 bool intersect_leaf_phicut( float4& isect, const quad& q0, const float t_min, const float3& o, const float3& d )
    146 {
    147     const float& cosPhi0 = q0.f.x ;
    148     const float& sinPhi0 = q0.f.y ;
    149     const float& cosPhi1 = q0.f.z ;
    150     const float& sinPhi1 = q0.f.w ;
    151 
    152     const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    153     bool zpure = d.x == 0.f && d.y == 0.f ;
    154     const float PR = cosPhi0*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    155     const float QR = cosPhi1*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi1  ;
    156     bool unbounded_exit = PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f )  ;
    157 

After zpure special case::

    EYE=0,0,2 UP=0,1,0  TMIN=0. CAM=1 ./cxr_geochain.sh 
        get expected cutaway quadrant 

    EYE=-2,0,1 TMIN=0. CAM=1 ./cxr_geochain.sh 
        expected full sphere from -X 

    EYE=-2,0,1 TMIN=0. CAM=1 ./cxr_geochain.sh 
        unexpected, no sign of cutaway 

    EYE=-2,0,4 TMIN=0. CAM=1 ./cxr_geochain.sh 
        still no sign of cutaway 

    EYE=1,1,1 TMIN=0. CAM=1 ./cxr_geochain.sh 
        cutaway evident : beach ball view

    EYE=-1,-1,1 TMIN=0. CAM=1 ./cxr_geochain.sh 
        dotty seam up middle 

    EYE=-1,-1,1 TMIN=0. CAM=0 ./cxr_geochain.sh 
        seam evident with perspective too

    EYE=-1,-1,1 TMIN=0. CAM=1 ZOOM=5 ./cxr_geochain.sh 
        zooming in on seam : it looks like a very regular dotted line 

Checking with imshow, looks like white line (misses) in the middle of the road::

    epsilon:CSGOptiX blyth$ DYDX=32,32 i tests/CSGOptiXRenderTest.py
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :       (65, 65, 4, 4) :  select the central portion of the image array  


Line of misses down the middle, one of the white road markings::

    epsilon:CSGOptiX blyth$ DYDX=1,1 i tests/CSGOptiXRenderTest.py
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) :  select the central portion of the image array  

    In [1]: b.view(np.int32)[:,:,3,3]                                                                                                                                                                    
    Out[1]: 
    array([[2, 1, 2],
           [2, 1, 2],
           [2, 1, 2]], dtype=int32)

Line of hits to the left::

    In [2]: b[:,0]                                                                                                                                                                                       
    Out[2]: 
    array([[[   0.211,    0.211,    0.789,    0.   ],
            [ -57.758,  -57.712,   57.735,    0.   ],
            [-100.023,  -99.977,  100.   ,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]],

           [[   0.211,    0.211,    0.789,    0.   ],
            [ -57.771,  -57.725,   57.709,    0.   ],
            [-100.036,  -99.99 ,   99.974,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]],

           [[   0.211,    0.211,    0.788,    0.   ],
            [ -57.784,  -57.739,   57.683,    0.   ],
            [-100.049, -100.004,   99.948,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]]], dtype=float32)


Line of misses in the middle::

    In [3]: b[:,1]                                                                                                                                                                                       
    Out[3]: 
    array([[[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-100.   , -100.   ,  100.   ,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]],

           [[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-100.013, -100.013,   99.974,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]],

           [[   1.   ,    1.   ,    1.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [-100.026, -100.026,   99.948,    1.732],
            [   0.577,    0.577,   -0.577,    0.   ]]], dtype=float32)


Hmm does not miss, need more precision ?:: 

    DUMP=3 ORI=-100,-100,100 DIR=1,1,-1 CSGQueryTest O

    epsilon:CSG blyth$ DUMP=3 ORI=-100,-100,100 DIR=1,1,-1 CSGQueryTest O
                               One HIT
                        q0 norm t (   -0.5774   -0.5774    0.5774   73.2051)
                       q1 ipos sd (  -57.7350  -57.7350   57.7350    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -100.0000 -100.0000  100.0000    0.0000)
                  q3 ray_dir gsid (    0.5774    0.5774   -0.5774 C4U (     0    0    0    0 ) )


Add a new LOAD "L" mode to CSGQueryTest that loads the isect subset written by CSGOptiX/tests/CSGOptiXRenderTest.py 
providing a way to rerun a pixel with exactly the same ray_origin, ray_direction and tmin using::

    epsilon:CSG blyth$ YX=1,1 CSGQueryTest
    2022-02-13 19:56:23.421 INFO  [35008083] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 1,1 )
                            noname MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -100.0131 -100.0131   99.9738    1.7321)
                  q3 ray_dir gsid (    0.5774    0.5774   -0.5774 C4U (     0    0    0    0 ) )


Rerunning the single pixel isect with debug::

    epsilon:CSG blyth$ YX=1,1 CSGQueryTest
    2022-02-13 19:59:14.356 INFO  [35012887] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 1,1 )
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -100.0131, -100.0131,   99.9738) 
    //intersect_leaf ray_direction (    0.5774,    0.5774,   -0.5774) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (    -0.5775    -0.5775     0.5771    73.2051)  
    //intersect_leaf valid_isect 1 isect (   -0.5775    -0.5775     0.5771    73.2051)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (   -0.5775    -0.5775     0.5771   -73.2051) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin ( -100.0131, -100.0131,   99.9738) 
    //intersect_leaf ray_direction (    0.5774,    0.5774,   -0.5774) 
    //intersect_leaf_phicut q0.f  (    0.0000     1.0000     1.0000     0.0000) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     1.7321 
    //intersect_leaf_phicut d.xyz (     0.5774     0.5774    -0.5774 ) zpure 0 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve:angle less than pi, -ve:angle greater than pi 
    //intersect_leaf_phicut PR    -0.5774 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR     0.5774 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 0 
    //intersect_leaf_phicut t_cand.0   173.2278 t_min     1.7321 
    //intersect_leaf_phicut ipos_x    -0.0000 ipos_x*1e6f    -7.6294  cosPhi0     0.0000  x_wrong_side 0 
    //intersect_leaf_phicut ipos_y    -0.0000 ipos_y*1e6f    -7.6294  sinPhi0     1.0000  y_wrong_side 1 
    //intersect_leaf_phicut t_cand   173.2278 t_min     1.7321 too_close 0 
    //intersect_leaf_phicut t_cand.1 999999988484154753734934528.0000 
    //intersect_leaf_phicut t_cand.2 999999988484154753734934528.0000 valid_intersect 0 
    //intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.0000     0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left    73.2051 t_right     0.0000 leftIsCloser 0  l_state Enter r_state  Miss l_cos*1e6f -1000000.0000 r_cos*1e6f     0.0000 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter    73.2051    r: Miss     0.0000     leftIsCloser 0 -> RETURN_MISS 
                            noname MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -100.0131 -100.0131   99.9738    1.7321)
                  q3 ray_dir gsid (    0.5774    0.5774   -0.5774 C4U (     0    0    0    0 ) )


Nextdoor pixel gives expected HIT::

    epsilon:CSG blyth$ YX=0,0 CSGQueryTest
    2022-02-13 20:04:55.216 INFO  [35017754] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 0,0 )
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -100.0227,  -99.9773,  100.0000) 
    //intersect_leaf ray_direction (    0.5774,    0.5774,   -0.5774) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (    -0.5776    -0.5771     0.5774    73.2051)  
    //intersect_leaf valid_isect 1 isect (   -0.5776    -0.5771     0.5774    73.2051)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (   -0.5776    -0.5771     0.5774   -73.2051) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin ( -100.0227,  -99.9773,  100.0000) 
    //intersect_leaf ray_direction (    0.5774,    0.5774,   -0.5774) 
    //intersect_leaf_phicut q0.f  (    0.0000     1.0000     1.0000     0.0000) cosPhi0/sinPhi0/cosPhi1/sinPhi1 t_min     1.7321 
    //intersect_leaf_phicut d.xyz (     0.5774     0.5774    -0.5774 ) zpure 0 
    //intersect_leaf_phicut PQ    -1.0000 cosPhi0*sinPhi1 - cosPhi1*sinPhi0 : +ve:angle less than pi, -ve:angle greater than pi 
    //intersect_leaf_phicut PR    -0.5774 cosPhi0*d.y - d.x*sinPhi0 
    //intersect_leaf_phicut QR     0.5774 cosPhi1*d.y - d.x*sinPhi1 
    //intersect_leaf_phicut unbounded_exit 0 
    //intersect_leaf_phicut t_cand.0   173.2444 t_min     1.7321 
    //intersect_leaf_phicut ipos_x     0.0000 ipos_x*1e6f     0.0000  cosPhi0     0.0000  x_wrong_side 0 
    //intersect_leaf_phicut ipos_y     0.0454 ipos_y*1e6f 45364.3789  sinPhi0     1.0000  y_wrong_side 0 
    //intersect_leaf_phicut t_cand   173.2444 t_min     1.7321 too_close 0 
    //intersect_leaf_phicut t_cand.1   173.2444 
    //intersect_leaf_phicut t_cand.2   173.2444 valid_intersect 1 
    //intersect_leaf valid_isect 1 isect (    1.0000     0.0000     0.0000   173.2444)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    1.0000     0.0000     0.0000   173.2444) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left    73.2051 t_right   173.2444 leftIsCloser 1  l_state Enter r_state  Exit l_cos*1e6f -1000000.0000 r_cos*1e6f 577350.2500 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter    73.2051    r: Exit   173.2444     leftIsCloser 1 -> RETURN_A 
                            noname HIT
                        q0 norm t (   -0.5776   -0.5771    0.5774   73.2051)
                       q1 ipos sd (  -57.7577  -57.7123   57.7350    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -100.0227  -99.9773  100.0000    1.7321)
                  q3 ray_dir gsid (    0.5774    0.5774   -0.5774 C4U (     0    0    0    0 ) )


::

    EYE=-1,-1,0 TMIN=0. CAM=1 ./cxr_geochain.sh
        very clear seam 

    EYE=1,1,0 TMIN=0. CAM=1 ./cxr_geochain.sh 
        beach ball view

    EYE=-1,-1,0 TMIN=0.7 CAM=1 ZOOM=2 ./cxr_geochain.sh 
        back view is informative using TMIN to cut a hole in the phere 
        showing the phicut planes : with a white line of misses up the middle  


Upside down Italian flag on the knife edge, from the two phicut faces and white line of misses up the sharp edge of phicut::

    epsilon:CSGOptiX blyth$ ./CSGOptiXRenderTest.sh 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) : /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy 





Start again with attempt to avoid two imprecise decisions at edge causing gap
-------------------------------------------------------------------------------

Now problem with normal incidence (1,0,0) direction::

    epsilon:CSG blyth$ SPURIOUS=1 ./csg_geochain.sh 

    SPURIOUS=1 IXYZ=-1,1,0  ./csg_geochain.sh 


Are missing the phicut at normal incidence and getting spurious hit on sphere::

    epsilon:CSG blyth$ DUMP=3 ORI=-1,1,0 DIR=1,0,0 CSGQueryTest O
                               One HIT
                        q0 norm t (    1.0000    0.0100    0.0000  100.9950)
                       q1 ipos sd (   99.9950    1.0000    0.0000    1.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (   -1.0000    1.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ DUMP=3 ORI=-1,10,0 DIR=1,0,0 CSGQueryTest O
                               One HIT
                        q0 norm t (    0.9950    0.1000    0.0000  100.4987)
                       q1 ipos sd (   99.4987   10.0000    0.0000   10.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (   -1.0000   10.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ DUMP=3 ORI=-1,50,0 DIR=1,0,0 CSGQueryTest O
                               One HIT
                        q0 norm t (    0.8660    0.5000    0.0000   87.6025)
                       q1 ipos sd (   86.6025   50.0000    0.0000   50.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (   -1.0000   50.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )


The problem is not normal incidence with phi0 plane, it is direction being parallel with phi1 plane
That causes t1 -inf so rearrange to make nan disqualify one root without killing the other::

    212     
    213     const float t0c = ( s0x <= 0.f || t0 <= t_min) ? RT_DEFAULT_MAX : t0 ;
    214     const float t1c = ( s1x <= 0.f || t1 <= t_min) ? RT_DEFAULT_MAX : t1 ;
    215     // comparisons with nan always give false, so make false correspond to disqualification  
    216     // t0, t1 can be -inf so must check them against t_min individually before comparing them 
    217     
    218     const float t_cand = fminf( t0c, t1c ) ;


After that::

    SPURIOUS=1 ./csg_geochain.sh

get spurious sphere intersects for +X rays on or beyond the pacmanpp phi0 plane

::

    epsilon:CSG blyth$ ORI=1,10,0 DIR=1,0,0 CSGQueryTest O
                               One HIT
                        q0 norm t (    0.9950    0.1000    0.0000   98.4987)
                       q1 ipos sd (   99.4987   10.0000    0.0000   10.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (    1.0000   10.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )


That is caused by false positive unbounded_exit for the face parallel rays. 

So add requiremeht that "sd <= 0" at the t_min point, so that requires on t_min point is within the phicut shape (or on surface):: 

    177     const float sd0 =  sinPhi0*(o.x+t_min*d.x) - cosPhi0*(o.y+t_min*d.y) ;
    178     const float sd1 = -sinPhi1*(o.x+t_min*d.x) + cosPhi1*(o.y+t_min*d.y) ;
    179     float sd = fminf( sd0, sd1 );   // signed distance at t_min 
    180     
    181 #ifdef DEBUG
    182     printf("//intersect_leaf_phicut  sd0 %10.4f sd1 %10.4f sd  %10.4f \n", sd0, sd1, sd ); 
    183 #endif
    184 
    185     
    186     const float PQ = cosPhi0*sinPhi1 - cosPhi1*sinPhi0  ;  // PQ +ve => angle < pi,   PQ -ve => angle > pi 
    187     const bool zpure = d.x == 0.f && d.y == 0.f ;   
    188     const float PR = cosPhi0*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi0  ;          // PR and QR +ve/-ve selects the "side of the line"
    189     const float QR = cosPhi1*(zpure ? o.y : d.y) - (zpure ? o.x : d.x)*sinPhi1  ;          
    190     const bool unbounded_exit = sd <= 0.f &&  ( PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f ))  ;
    191     

Which eliminates most spurious, leaving 6 all starting on the phi0 plane and heading +X::

      SPURIOUS=1 ./csg_geochain.sh

      s_isect_gsid (   0   1   0   0 )   s_t (    98.8686 )   s_pos (    98.8686    15.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    15.0000 )  
      s_isect_gsid (   0   2   0   0 )   s_t (    95.3939 )   s_pos (    95.3939    30.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    30.0000 )  
      s_isect_gsid (   0   3   0   0 )   s_t (    89.3029 )   s_pos (    89.3029    45.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    45.0000 )  
      s_isect_gsid (   0   4   0   0 )   s_t (    80.0000 )   s_pos (    80.0000    60.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    60.0000 )  
      s_isect_gsid (   0   5   0   0 )   s_t (    66.1438 )   s_pos (    66.1438    75.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    66.1438 )  
      s_isect_gsid (   0   6   0   0 )   s_t (    43.5890 )   s_pos (    43.5890    90.0000     0.0000 )   s_pos_r (   100.0000 )   s_sd (    43.5890 )  

Modify the requirement to not allow sd==0.f gets rid of all SPURIOUS::

     const bool unbounded_exit = sd < 0.f &&  ( PQ >= 0.f ? ( PR >= 0.f && QR <= 0.f ) : ( PR >= 0.f || QR <= 0.f ))  ; 


Looking again at cxr are back to previous situation with the edge seam::

    EYE=-1,-1,0 TMIN=0.7 CAM=1 ./cxr_geochain.sh 
        inverted italian flag   

    EYE=1,0,0 CAM=1 ./cxr_geochain.sh 
        right side hemi disappeared

        * default tmin of 1 is suprisingly large : corresponding to extent of solid : in order to chop in half 
        * this means that disappeating the phicut face is actually expected behaviour  

    EYE=1,0,0 CAM=1 TMIN=0.99 ./cxr_geochain.sh
         no longer disappeared right hemi because pull back the tmin 

    EYE=1.1,0,0 CAM=1 ./cxr_geochain.sh 
        right side hemi appears when move back 

    EYE=1,0,0 CAM=1 TMIN=0 ./cxr_geochain.sh
        again appears when set TMIN zero 

    EYE=2,0,0 CAM=0 ./cxr_geochain.sh 
        smaller right hemi : not expected 

        * actually : i think it is expected, it just looks a bit funny due to the perfection of the alignment  

    EYE=2,0,0 CAM=1 ./cxr_geochain.sh 
        looks correct other than seam line

        * still with seam even after "one decision on knife edge"
        * seam removed by allowing zero 

    EYE=2,0.1,0 CAM=1 ./cxr_geochain.sh 
        * from slightly to the right +Y no seam, expected sliver of flat face 

    EYE=2,2,0 CAM=1 ./cxr_geochain.sh 
        beach ball view, looks as expected : no seam  

    EYE=0,0,2 UP=0,1,0  CAM=1 TMIN=0 ./cxr_geochain.sh 
         expected cut quadrant 

    EYE=0,2,0 CAM=1 TMIN=0 ./cxr_geochain.sh 
         unexpected flat : means get intersecting with the plane even when should not 

         * no longer flat

    PARADIR=-1,0,0 PHO=-10 ./csg_geochain.sh 
         DONE: added PARADIR parallel mode to csg_geochain.sh that overrides photon directions

    EYE=0.1,2,0 CAM=1 TMIN=0 ./cxr_geochain.sh
         shifting a little, the curved hemi appears on right 

    EYE=-0.1,2,0 CAM=1 TMIN=0 ./cxr_geochain.sh
         shifting a little the other way, curved hemi still there bit now with seam 

         * no more seam 


Look into the seam::

    EYE=2,0,0 CAM=1 ZOOM=5 ./cxr_geochain.sh 

Save the central pixels::   

    epsilon:CSGOptiX blyth$ ./CSGOptiXRenderTest.sh 
    a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
    b :         (3, 3, 4, 4) : /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy 

Reproduce the center line seam of MISS::

    epsilon:CSG blyth$ ./CSGQueryTest.sh 
    2022-02-14 20:34:35.717 INFO  [35931039] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 1,1 )
                            noname MISS
                        q0 norm t (    0.0000    0.0000    0.0000    0.0000)
                       q1 ipos sd (    0.0000    0.0000    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  200.0000    0.0000   -0.0741  100.0000)
                  q3 ray_dir gsid (   -1.0000    0.0000    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ 


Allowing zero gets rid of the seam.




Find another seam : looks to be a picking the axis line intersect seam, not from MISS::

    EYE=-2,2,2 CAM=1 TMIN=0 ./cxr_geochain.sh 
    EYE=-2,2,0 CAM=1 TMIN=0 ./cxr_geochain.sh 


    (-2,2)
      +    Y
        .  |
          .|
     - - - +----X

    EYE=-2,2,0 CAM=1 TMIN=0 ZOOM=10 ./cxr_geochain.sh

Confirm not miss, its different shade of green::

    epsilon:CSGOptiX blyth$ ./CSGOptiXRenderTest.sh 
       a :   (1080, 1920, 4, 4) : /tmp/blyth/opticks/GeoChain_Darwin/GeneralSphereDEV/CSGOptiXRenderTest/cvd0/50001/ALL/top_i0_/cxr_geochain_GeneralSphereDEV_ALL_isect.npy 
       b :         (3, 3, 4, 4) : /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy 

        
    epsilon:CSG blyth$ YX=1,1 ./CSGQueryTest.sh 
    2022-02-14 20:57:24.838 INFO  [35966184] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 1,1 )
                            noname HIT
                        q0 norm t (    0.0000    1.0000    0.0000  282.8427)
                       q1 ipos sd (    0.0000    0.0000   -0.0524    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -200.0000  200.0000   -0.0524    2.8284)
                  q3 ray_dir gsid (    0.7071   -0.7071    0.0000 C4U (     0    0    0    0 ) )

    epsilon:CSG blyth$ YX=0,0 ./CSGQueryTest.sh 
    2022-02-14 20:57:39.008 INFO  [35966561] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 0,0 )
                            noname HIT
                        q0 norm t (   -0.7067    0.7075    0.0000  182.8427)
                       q1 ipos sd (  -70.6737   70.7477    0.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -199.9630  200.0370    0.0000    2.8284)
                  q3 ray_dir gsid (    0.7071   -0.7071    0.0000 C4U (     0    0    0    0 ) )



::

    epsilon:CSG blyth$ YX=1,1 ./CSGQueryTest.sh 
    2022-02-14 21:14:55.378 INFO  [35983347] [CSGQueryTest::Load@288]  a (3, 3, 4, 4, ) LOAD loadpath /tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy YX ( 1,1 )
    //intersect_prim
    //intersect_tree  numNode 3 height 1 fullTree(hex) 20000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -200.0000,  200.0000,   -0.0524) 
    //intersect_leaf ray_direction (    0.7071,   -0.7071,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (    -0.7071     0.7071    -0.0005   182.8427)  
    //intersect_leaf valid_isect 1 isect (   -0.7071     0.7071    -0.0005   182.8427)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (   -0.7071     0.7071    -0.0005  -182.8427) 
    //intersect_tree  nodeIdx 3 CSG::Name     phicut depth 1 elevation 0 
    //intersect_tree  nodeIdx 3 primitive 1 
    //intersect_leaf typecode 120 gtransformIdx 2 
    //intersect_leaf ray_origin ( -200.0000,  200.0000,   -0.0524) 
    //intersect_leaf ray_direction (    0.7071,   -0.7071,    0.0000) 
    //intersect_leaf_phicut  q0.f (    0.0000     1.0000       1.0000     0.0000 )   cosPhi0 sinPhi0   cosPhi1 sinPhi1      t_min      2.8284 
    //intersect_leaf_phicut  sd0  -198.0000 sd1   198.0000 sd   -198.0000 
    //intersect_leaf_phicut  PQ    -1.0000 zpure 0 
    //intersect_leaf_phicut  PR    -0.7071  QR    -0.7071 unbounded_exit 1 
    //intersect_leaf_phicut  t0   282.8427  (x0,y0) (    0.0000,     0.0000)   (s0x,s0y) (         0,          0) 
    //intersect_leaf_phicut  t1   282.8427  (x1,y1) (    0.0000,     0.0000)   (s1x,s1y) (    0.0000,     0.0000) 
    //intersect_leaf_phicut t0c   282.8427 t1c   282.8427 safezone 0 t_cand   282.8427 valid_intersect 1  unbounded_exit 1 
    //intersect_leaf valid_isect 1 isect (    0.0000     1.0000     0.0000   282.8427)   
    //intersect_tree  nodeIdx 3 primitive 1 nd_isect (    0.0000     1.0000     0.0000   282.8427) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left   182.8427 t_right   282.8427 leftIsCloser 1  l_state Enter r_state Enter l_cos*1e6f -1000000.2500 r_cos*1e6f -707106.7500 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter   182.8427    r:Enter   282.8427     leftIsCloser 1 -> LOOP_A 
    //intersect_tree nodeIdx  1 height  1 depth  0 elevation  1 endTree    10000 leftTree  3020000 rightTree  1030000 
    //intersect_tree  nodeIdx 2 CSG::Name     sphere depth 1 elevation 0 
    //intersect_tree  nodeIdx 2 primitive 1 
    //intersect_leaf typecode 101 gtransformIdx 1 
    //intersect_leaf ray_origin ( -200.0000,  200.0000,   -0.0524) 
    //intersect_leaf ray_direction (    0.7071,   -0.7071,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (     0.7071    -0.7071    -0.0005   382.8428)  
    //intersect_leaf valid_isect 1 isect (    0.7071    -0.7071    -0.0005   382.8428)   
    //intersect_tree  nodeIdx 2 primitive 1 nd_isect (    0.7071    -0.7071    -0.0005  -382.8428) 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 primitive 0 

    //intersect_tree nodeIdx   1 t_left   382.8428 t_right   282.8427 leftIsCloser 0  l_state  Exit r_state Enter l_cos*1e6f 1000000.6250 r_cos*1e6f -707106.7500 
    //   1 : stack peeking : left 1 right 0 (stackIdx)     intersection  l: Exit   382.8428    r:Enter   282.8427     leftIsCloser 0 -> RETURN_B 
                            noname HIT
                        q0 norm t (    0.0000    1.0000    0.0000  282.8427)
                       q1 ipos sd (    0.0000    0.0000   -0.0524    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -200.0000  200.0000   -0.0524    2.8284)
                  q3 ray_dir gsid (    0.7071   -0.7071    0.0000 C4U (     0    0    0    0 ) )


