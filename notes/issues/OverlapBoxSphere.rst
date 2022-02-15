OverlapBoxSphere
==================

Fixed Corner clipper spurious
--------------------------------

Corner clippers giving spurious::

   SPURIOUS=1 ./csg_geochain.sh 

   SPURIOUS=1 IXYZ=-15,0 ./csg_geochain.sh 


1. enters and exits box crossing a corner and then intersects the sphere
               

Select single photon corner clipper and rerun::

     SXYZW=-15,0,0,7 GEOM=OverlapBoxSphere_XY ./csg_geochain.sh run 


This should yield a MISS::
              
    2022-02-15 17:13:52.657 INFO  [257488] [CSGGeometry::saveCenterExtentGenstepIntersect@196] [ pp.size 62700 t_min     0.0000
    2022-02-15 17:13:52.657 INFO  [257488] [CSGGeometry::saveCenterExtentGenstepIntersect@225] [ single photon selected
    //intersect_prim typecode 13 
    //intersect_leaf typecode 110 gtransformIdx 0 
    //intersect_leaf ray_origin ( -225.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.9029,    0.4298,    0.0000) 
    //intersect_leaf_box3  bmin (  -75.0000,  -75.0000,  -75.0000) bmax (   75.0000,   75.0000,   75.0000)  
    //intersect_leaf_box3  along_xyz (0,0,0) in_xyz (0,1,1)   has_intersect 1  
    //intersect_leaf_box3 t_min     0.0000 t_near   166.1265 t_far   174.5018 t_cand   166.1265 
    //intersect_leaf_box3 has_valid_intersect 1  isect (    -1.0000     0.0000     0.0000   166.1265)  
    //intersect_leaf valid_isect 1 isect (   -1.0000     0.0000     0.0000   166.1265)   
    //intersect_leaf typecode 101 gtransformIdx 0 
    //intersect_leaf ray_origin ( -225.0000,    0.0000,    0.0000) 
    //intersect_leaf ray_direction (    0.9029,    0.4298,    0.0000) 
    //intersect_leaf_sphere radius   100.0000 
    //intersect_leaf_sphere valid_isect 1  isect (    -0.6455     0.7637     0.0000   177.6955)  
    //intersect_leaf valid_isect 1 isect (   -0.6455     0.7637     0.0000   177.6955)   
    //intersect_node_overlap num_sub 2  enter_count 2 exit_count 0 valid_isect 1 isect_farthest_enter.w   177.6955 isect_nearest_exit.w 999999988484154753734934528.0000 
    //distance_node_list isub 0 sub_sd     1.3726 sd     1.3726 
     //distance_node_list isub 1 sub_sd     0.0000 sd     1.3726 
     2022-02-15 17:13:52.657 INFO  [257488] [CSGGeometry::saveCenterExtentGenstepIntersect@228] 
            single photon selected HIT
                        q0 norm t (   -0.6455    0.7637    0.0000  177.6955)
                       q1 ipos sd (  -64.5540   76.3726    0.0000    1.3726)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -225.0000    0.0000    0.0000    0.0000)
                  q3 ray_dir gsid (    0.9029    0.4298    0.0000 C4U (   -15    0    0    7 ) )





Rays only intersecting the sphere or box and not both, giving hits


SPURIOUS=1 IXYZ=-16,9,0 ./csg_geochain.sh ana

SPURIOUS=1 IXYZ=16,8,0 ./csg_geochain.sh ana

SPURIOUS=1 IXYZ=16,8,0 GEOM=OverlapBoxSphere_XY ./csg_geochain.sh 


                               count_all : 24422 
                          count_spurious : 4104 
                        count_isect_gsid : 13   IXYZ 16,8,0  ix:16 iy:8 iz:0
                            count_select : 13  

                          count_spurious : 4104    SPURIOUS 1 
                            count_select : 2  

                            count_select : 2  

                                 s_count : 2  

                               s_limited : 2  

                          selected_isect : 2  

  s_isect_gsid (  16   8   0  51 )   s_t (   224.6276 )   s_pos (    16.3896    98.6478     0.0000 )   s_pos_r (   100.0000 )   s_sd (    23.6478 )  
  s_isect_gsid (  16   8   0  63 )   s_t (   251.9620 )   s_pos (    75.0000   -70.4202     0.0000 )   s_pos_r (   102.8786 )   s_sd (     2.8786 )  
 Use IXYZ to select gensteps, IW to select photons within the genstep 
INFO:__main__:asd_cut      0.001 sd.min -3.052e-05 sd.max         25 num select 2 
 define key SAVE_SELECTED_ISECT to save selected isect to /tmp/selected_isect.npy 
ReferenceGeometry for GEOM OverlapBoxSphere_XY 
ReferenceGeometry not implemented for GEOM OverlapBoxSphere_XY 
 outpath : /tmp/blyth/opticks/GeoChain_Darwin/OverlapBoxSphere/CSGIntersectSolidTest/OverlapBoxSphere_XY/figs/out.png 



Rerun one ray 

SXYZW=16,8,0,51 ./csg_geochain.sh run


2022-02-15 19:55:21.197 INFO  [328295] [CSGGeometry::saveCenterExtentGenstepIntersect@196] [ pp.size 62700 t_min     0.0000
2022-02-15 19:55:21.197 INFO  [328295] [CSGGeometry::saveCenterExtentGenstepIntersect@225] [ single photon selected
//intersect_prim typecode 13 
//intersect_leaf typecode 110 gtransformIdx 0 
//intersect_leaf ray_origin (  240.0000,  120.0000,    0.0000) 
//intersect_leaf ray_direction (   -0.9955,   -0.0951,    0.0000) 
//intersect_leaf_box3  bmin (  -75.0000,  -75.0000,  -75.0000) bmax (   75.0000,   75.0000,   75.0000)  
//intersect_leaf_box3  along_xyz (0,0,0) in_xyz (0,0,1)   has_intersect 0  
//intersect_leaf_box3 has_valid_intersect 0  isect (     0.0000     0.0000     0.0000     0.0000)  
//intersect_leaf valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
//intersect_leaf typecode 101 gtransformIdx 0 
//intersect_leaf ray_origin (  240.0000,  120.0000,    0.0000) 
//intersect_leaf ray_direction (   -0.9955,   -0.0951,    0.0000) 
//intersect_leaf_sphere radius   100.0000 
//intersect_leaf_sphere valid_isect 1  isect (     0.1639     0.9865     0.0000   224.6276)  
//intersect_leaf valid_isect 1 isect (    0.1639     0.9865     0.0000   224.6276)   
//intersect_leaf typecode 101 gtransformIdx 0 
//intersect_leaf ray_origin (  240.0000,  120.0000,    0.0000) 
//intersect_leaf ray_direction (   -0.9955,   -0.0951,    0.0000) 
//intersect_leaf_sphere radius   100.0000 
//intersect_leaf_sphere valid_isect 1  isect (    -0.3476     0.9376     0.0000   276.0125)  
//intersect_leaf valid_isect 1 isect (   -0.3476     0.9376     0.0000   276.0125)   
//intersect_node_overlap num_sub 2  enter_count 1 exit_count 1 valid_isect 1 isect_farthest_enter.w   224.6276 isect_nearest_exit.w   276.0125 
//distance_node_list isub 0 sub_sd    23.6478 sd    23.6478 
 //distance_node_list isub 1 sub_sd     0.0000 sd    23.6478 
 2022-02-15 19:55:21.197 INFO  [328295] [CSGGeometry::saveCenterExtentGenstepIntersect@228] 
        single photon selected HIT
                    q0 norm t (    0.1639    0.9865    0.0000  224.6276)
                   q1 ipos sd (   16.3896   98.6478    0.0000   23.6478)- sd < SD_CUT :    -0.0010
             q2 ray_ori t_min (  240.0000  120.0000    0.0000    0.0000)
              q3 ray_dir gsid (   -0.9955   -0.0951    0.0000 C4U (    16    8    0   51 ) )

2022-02-15 19:55:21.197 INFO  [328295] [CSGGeometry::saveCenterExtentGenstepIntersect@229] ] single photon selected 



                                
                          +----------------------+
                          |                      |
              0- - - - - -1 - - - - - - - - - - -2   
                          E                      X
                          |                      |
                          |                      |
                +---------|----------------+     |
                |         | . . . . . . . .|     |
                |         | . . . . . . . .|     |
          0- - -1 - -  - -2 - -  - - - -  -3 - - 4
                E         E . . . . . . . .X     X
                |         | . . . . . . . .|     |
                |         | . . . . . . . .|     |
                |   0 - - 1 - - - - - - - -2 - - 3
                |         E . . . . . . . .X     X
                |         | . . . . . . . .|     |
                |         | . . . . . . . .|     |
                |         | . . .0 - - - - 1 - - 2
                |         | . . . . . . . .X     X
                |         | . . . . . . . .|     |
                |         +----------------|-----+
                |                          |
                |                          |
                +---------------------------+


                

Looks OK, tmin cutting seems to work::

    EYE=1,1,1 TMIN=0 ./cxr_geochain.sh 



