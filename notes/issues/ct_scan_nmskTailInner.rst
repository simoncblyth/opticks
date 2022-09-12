ct_scan_nmskTailInner
========================


nmskTailOuterITube and nmskTailOuterITube : checkz has peak at expected place but large cloud
------------------------------------------------------------------------------------------------

::

    In [1]: mpplt_hist( mp, np.abs(d[:,5,3]), bins=50 )   


with fat cylinder nmskTailOuterIITube : the checkz is as expected
--------------------------------------------------------------------


check with fat cylinder::

    geom_  # nmskTailOuterIITube
    c
    ./ct.sh 

    In [1]: mpplt_hist( mp, np.abs(d[:,5,3]) )   ## checkz 

    In [2]: np.abs(d[:,5,3])
    Out[2]: 
    array([72.099, 72.115, 72.128, 72.095, 72.104, 72.12 , 72.105, 72.111, 72.111, 72.125, 72.104, 72.11 , 72.114, 72.115, 72.116, 72.112, 72.113, 72.114, 72.118, 72.108, 72.113, 72.103, 72.116, 72.104,
           72.122, 72.106, 72.117, 72.097, 72.104, 72.099, 72.112, 72.097, 72.09 , 72.09 , 72.095, 72.127, 72.123], dtype=float32)

    In [3]: np.abs(d[:,5,3]).min()
    Out[3]: 72.09011

    In [4]: np.abs(d[:,5,3]).max()
    Out[4]: 72.12756



indep endcap intersect
------------------------


::

     856 /**
     857 intersect_leaf_plane
     858 -----------------------
     859 
     860 * https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
     861 
     862 Equation for points p that are in the plane::
     863 
     864    (p - p0).n = 0      
     865 
     866    p0   : point in plane which is pointed to by normal n vector from origin,  
     867    p-p0 : vector that lies within the plane, and hence is perpendicular to the normal direction 
     868    p0.n : d, distance from plane to origin 
     869 
     870 
     871    p = o + t v   : parametric ray equation  
     872 
     873    (o + t v - p0).n = 0 
     874 
     875    (p0-o).n  = t v.n
     876 
     877             (p0 - o).n        d - o.n
     878        t  = -----------  =   -----------
     879                v.n              v.n  
     880 **/


::

     +---------------------------+--------I------------------+   z2
     |                           |        :                  |
     +                           +        :                  +
     |                           |        O                  |
     +---------------------------+---------------------------+   z1


     n2 = [0,0,1] 
     
            t2 = ( z2 - ray_origin.z )/ray_direction.z  
 
            ray_origin.x*ray_origin.x + ray_origin.y*ray_origin.y > rr 

            t2 = z2 - ray_origin.z 
            t1 = z1 - ray_origin.z  




::

    1093     // axial ray endcap handling : can treat axial rays in 2d way 
    1094     if(fabs(a) < 1e-6f)
    1095     {
    1096 
    1097 #ifdef DEBUG_RECORD
    1098     printf("//intersect_leaf_cylinder : axial ray endcap handling, a %10.4g c(dd*k - md*md) %10.4g dd %10.4g k %10.4g md %10.4g  \n", a, c,dd,k,md );
    1099 #endif
    1100         if(c > 0.f) return false ;  // ray starts and ends outside cylinder
    1101 
    1102         float t_PCAP_AX = -mn/nn  ;
    1103         float t_QCAP_AX = (nd - mn)/nn ;

    /// problem not only edges so must be precision loss on these t ?  BUT nn is 1. 


Can simply do a checkz on the candidate intersect ?::

    In [32]: d[0,0,:3]+d[0,7,3]*d[0,1,:3]
    Out[32]: array([-264.155,    0.   ,   -0.323], dtype=float32)




    1104 
    1105         if(md < 0.f )     // ray origin on P side
    1106         {
    1107             t_cand = t_PCAP_AX > t_min ? t_PCAP_AX : t_QCAP_AX ;

    /// HMM: maybe should disqualify the root by setting it to t_min ? no both roots should be in play as t_min could disqualify one 

    1108         }
    1109         else if(md > dd )  // ray origin on Q side 
    1110         {
    1111             t_cand = t_QCAP_AX > t_min ? t_QCAP_AX : t_PCAP_AX ;
    1112         }
    1113         else              // ray origin inside,   nd > 0 ray along +d towards Q  
    1114         {
    1115             t_cand = nd > 0.f ? t_QCAP_AX : t_PCAP_AX ;
    1116         }
    1117 
    1118         unsigned endcap = t_cand == t_PCAP_AX ? ENDCAP_P : ( t_cand == t_QCAP_AX ? ENDCAP_Q : 0u ) ;
    1119    




hmm : a simpler ray-cylinder intersection func would be good
---------------------------------------------------------------

The below approach looks nice but it doesnt handle the endcaps and axial rays 
which are giving the trouble. 



* https://math.stackexchange.com/questions/3248356/calculating-ray-cylinder-intersection-points


The points at which the ray intersects the cylinder are the only ones on the
line that are at a distance equal to the radius from the cylinder’s axis. Since
you’re starting from a description of the cylinder as axis and radius, you can
use a standard formula for the distance from a point to a line to find these
points instead of trying to come up with an equation for the cylinder or trying
to come up with a transformation into some standard configuration.

Let 𝐱(𝑡)=𝐩0+𝑡𝐯

be the parameterization of the ray with the given starting point and direction
vector. Choose two points 𝐱1 and 𝐱2 on the cylinder’s axis: since that’s also
defined by a ray (line?) you can choose the origin point of that line for 𝐱1
and add any convenient multiple of the axis direction vector to it for the
other. Letting 𝑟 be the cylinder’s radius, the point-line distance formula
gives following the quadratic equation in 𝑡: |(𝐱(𝑡)−𝐱1)×(𝐱(𝑡)−𝐱2)|2|𝐱1−𝐱2|2=𝑟2.

Expand and solve for 𝑡, rejecting any negative solutions, then compute 𝐱(𝑡) for
each resulting value of 𝑡. The one with the lesser 𝑡-value is the nearer to the origin of the ray.

For a finite cylinder, you can then project these points onto the cylinder’s
axis and perform a range check. If you choose for 𝐱1 and 𝐱2 above the two
points on the cylinder’s axis that bound the cylinder, then if 𝐩 is a solution
to the infinite intersection, it lies on the bounded cylinder iff
0≤(𝐩−𝐱1)⋅(𝐱2−𝐱1)≤(𝐱2−𝐱1)⋅(𝐱2−𝐱1).



issue 2 : manifests with nmskTailOuterITube hz 0.15 mm alone with regularly spaced spills along the length of the cylinder
---------------------------------------------------------------------------------------------------------------------------

* the regularity could simply be from where the genstep sources are 

* HMM: ARE THEY FROM AXIAL RAYS ?  YES : ALL 227 SELECTED BELOW ARE +-Z DIRECTION RAYS

* in 3D those are presumably some kind of float precision artifact rings 
* testing with nmskTailInnerITube__U1 hz 0.65 mm shows a very small amount of spill at the ends, 
  suggesting the problem gets worse as the cylinder gets thinner 

* SO THE PROBLEM LOOKS TO BE CAUSED BY PRECISION LOSS IN VERY THIN CYLINDER INTERSECTION 
 
  * AND IT APPEARS TO BE IN THE AXIAL SPECIAL CASE 
  * COLLECTED AXIAL CALC INTERMEDIATES USING CSGDebug_Cylinder


::

    GEOM=nmskTailOuterITube__U1 ./ct.sh 

    In [9]: w = np.abs(s_pos[:,2]) > 0.15 + 0.01

    In [10]: s_pos[w]                                                                                                                                                           
    Out[10]: 
    array([[-264.155,    0.   ,   -0.323],
           [-264.398,    0.   ,   -0.722],
           [-263.616,    0.   ,    0.397],
           [-263.965,    0.   ,    0.195],
           [-263.935,    0.   ,    0.485],
           [-264.199,    0.   ,    0.819],
           [-264.222,    0.   ,    0.708],
           [-264.071,    0.   ,    0.329],
           [-264.345,    0.   ,    0.728],
           [-263.656,    0.   ,   -0.232],
           [-263.319,    0.   ,   -0.602],
           [-264.239,    0.   ,    0.417],
           [-237.854,    0.   ,   -0.477],
           [-237.078,    0.   ,    0.628],
           [-237.331,    0.   ,    0.252],
           [-237.388,    0.   ,    0.324],
           [-237.656,    0.   ,   -0.318],
           [-237.503,    0.   ,    0.286],
           [-237.745,    0.   ,   -0.813],
           [-237.539,    0.   ,    0.389],
           [-237.607,    0.   ,    0.217],
           [-237.64 ,    0.   ,    0.519],
           [-237.649,    0.   ,    0.602],

    In [12]: len(s_pos[w])                                                                                                                                                      
    Out[12]: 227


simpler to select on original array indices::

    In [20]: np.abs(s.simtrace[:,1,2]) > 0.16                                                                                                                                   
    Out[20]: array([False, False, False, False, False, ..., False, False, False, False, False])

    In [21]: w = np.abs(s.simtrace[:,1,2]) > 0.16                                                                                                                               

    In [23]: s.simtrace[w].shape                                                                                                                                                
    Out[23]: (227, 4, 4)


All the spill come from near axial rays, so it is an axial ray problem::

    In [25]: s.simtrace[w,3,:3]                                                                                                                                                 
    Out[25]: 
    array([[-0.001,  0.   ,  1.   ],
           [-0.002,  0.   ,  1.   ],
           [ 0.002,  0.   ,  1.   ],
           [ 0.001,  0.   ,  1.   ],
           [ 0.002,  0.   ,  1.   ],
           [-0.003,  0.   , -1.   ],
           [-0.002,  0.   , -1.   ],
           [-0.001,  0.   , -1.   ],
           [-0.002,  0.   , -1.   ],
           [ 0.001,  0.   , -1.   ],
           [ 0.003,  0.   , -1.   ],
           [-0.001,  0.   , -1.   ],
           [-0.001,  0.   ,  1.   ],





issue 2 : "spill" off ends of the sub-mm lips from ~vertical/horizontal rays
-----------------------------------------------------------------------------------

* added selection handling to CSG/ct.sh to look into this
* rogue intersects have +z/-z normals : would guess that the v.thin cylinders are implicated

::

    In [5]: sts[:,:,:3]
    Out[5]: 
    array([[[   0.   ,    0.   ,   -1.   ],        
            [ 264.525,    0.   ,  -40.112],
            [ 264.   ,    0.   , -211.2  ],
            [   0.003,    0.   ,    1.   ]],   ## +Z dir 

           [[   0.   ,    0.   ,    1.   ],
            [ 264.84 ,    0.   ,  -38.194],
            [ 264.   ,    0.   ,  237.6  ],
            [   0.003,    0.   ,   -1.   ]]], dtype=float32)     ## -Z dir

    In [8]: np.where(w)[0]
    Out[8]: array([495871, 512880])




::

    2022-09-12 14:51:29.931 INFO  [4293206] [CSGQuery::init@65]  sopr 0:0 solidIdx 0 primIdxRel 0
    NP::init size 16 ebyte 4 num_char 64
    2022-09-12 14:51:29.932 INFO  [4293206] [CSGDraw::draw@57] CSGSimtrace axis Z
    2022-09-12 14:51:29.932 INFO  [4293206] [CSGDraw::draw@58]  type 2 CSG::Name(type) intersection IsTree 1 width 15 height 3

                                                       in                                                                                                                     
                                                      1                                                                                                                       
                                                         0.00                                                                                                                 
                                                        -0.00                                                                                                                 
                                                                                                                                                                              
                                   un                                                          in                                                                             
                                  2                                                           3                                                                               
                                     0.00                                                        0.00                                                                         
                                    -0.00                                                       -0.00                                                                         
                                                                                                                                                                              
               un                            cy                            in                           !cy                                                                   
              4                             5                             6                             7                                                                     
                 0.00                        -39.00                          0.00                        -39.00                                                               
                -0.00                       -183.22                         -0.00                       -175.22                                                               
                                                                                                                                                                              
     zs                  cy                                     !zs                 !cy                                                                                       
    8                   9                                       12                  13                                                                                        
     -39.00              -39.00                                  -39.00              -38.00                                                                                   
    -194.10              -39.30                                 -186.10              -39.30                                                                                   
                                                                                                                                                                              
                                                                                                                                                                              
                                                                                                                                                                              
                                                                                                                                                                              
                                                                                                                                                                              
                                                                                                                                                                              
    2022-09-12 14:51:29.932 INFO  [4293206] [CSGSimtrace::init@44]  frame.ce ( 0.000, 0.000,-97.050,264.000)  SELECTION 495871 num_selection 1
    2022-09-12 14:51:29.932 INFO  [4293206] [SFrameGenstep::StandardizeCEGS@437]  CEGS  ix0 ix1 -16 16 iy0 iy1 0 0 iz0 iz1 -9 9 photons_per_genstep 1000 grid_points (ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) 627 tot_photons (grid_points*photons_per_genstep) 627000
    2022-09-12 14:51:29.932 INFO  [4293206] [SFrameGenstep::GetGridConfig@111]  ekey CEGS Desc  size 8[-16 16 0 0 -9 9 1000 1 ]
    2022-09-12 14:51:29.932 INFO  [4293206] [SFrameGenstep::CE_OFFSET@68] ekey CE_OFFSET val (null) is_CE 0 ce_offset.size 1 ce ( 0.000, 0.000,-97.050,264.000) 
    SFrameGenstep::Desc ce_offset.size 1
       0 : ( 0.000, 0.000, 0.000) 

    2022-09-12 14:51:29.932 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@146]  ce ( 0.000, 0.000,-97.050,264.000)  ce_offset.size 1
    2022-09-12 14:51:29.932 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@287]  num_offset 1 ce_scale 1 nx 16 ny 0 nz 9 GridAxes 2 GridAxesName XZ high 1 gridscale 0.1 scale 0.1
    2022-09-12 14:51:29.937 INFO  [4293206] [SFrameGenstep::GetGridConfig@111]  ekey CEHIGH_0 Desc  size 0[]
    2022-09-12 14:51:29.937 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@171]  key CEHIGH_0 cehigh.size 0
    2022-09-12 14:51:29.937 INFO  [4293206] [SFrameGenstep::GetGridConfig@111]  ekey CEHIGH_1 Desc  size 0[]
    2022-09-12 14:51:29.937 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@171]  key CEHIGH_1 cehigh.size 0
    2022-09-12 14:51:29.937 INFO  [4293206] [SFrameGenstep::GetGridConfig@111]  ekey CEHIGH_2 Desc  size 0[]
    2022-09-12 14:51:29.937 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@171]  key CEHIGH_2 cehigh.size 0
    2022-09-12 14:51:29.937 INFO  [4293206] [SFrameGenstep::GetGridConfig@111]  ekey CEHIGH_3 Desc  size 0[]
    2022-09-12 14:51:29.937 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@171]  key CEHIGH_3 cehigh.size 0
    2022-09-12 14:51:29.937 INFO  [4293206] [*SFrameGenstep::MakeCenterExtentGensteps@179]  gsl.size 1
      0 NP  dtype <f4(627, 6, 4, ) size 15048 uifc f ebyte 4 shape.size 3 data.size 60192 meta.size 0 names.size 0 nv 24
     ni_total 627
     c NP  dtype <f4(627, 6, 4, ) size 15048 uifc f ebyte 4 shape.size 3 data.size 60192 meta.size 0 names.size 0
    2022-09-12 14:51:29.941 ERROR [4293206] [SEvt::setFrame_HostsideSimtrace@306] frame.is_hostside_simtrace num_photon_gs 627000 num_photon_evt 627000
    2022-09-12 14:51:29.941 INFO  [4293206] [SEvt::setFrame_HostsideSimtrace@315]  before hostside_running_resize simtrace.size 0
    2022-09-12 14:51:30.002 INFO  [4293206] [SEvt::setFrame_HostsideSimtrace@319]  after hostside_running_resize simtrace.size 627000
    2022-09-12 14:51:30.003 ERROR [4293206] [SFrameGenstep::GenerateSimtracePhotons@675] SFrameGenstep::GenerateSimtracePhotons simtrace.size 627000
    2022-09-12 14:51:30.111 INFO  [4293206] [SFrameGenstep::GenerateSimtracePhotons@760]  simtrace.size 627000
    //intersect_prim typecode 2 name intersection 
    //intersect_tree  numNode(subNum) 15 height 3 fullTree(hex) 80000 
    //intersect_tree  nodeIdx 8 CSG::Name    zsphere depth 3 elevation 0 
    //intersect_tree  nodeIdx 8 node_or_leaf 1 
    //intersect_node typecode 103 name zsphere 
    //[intersect_leaf typecode 103 name zsphere gtransformIdx 3 
    //[intersect_leaf_zsphere radius   194.0000 b  -210.7613 c 44605.4375 
    // intersect_leaf_zsphere radius   194.0000 zmax   -39.0000 zmin  -194.1000  with_upper_cut 1 with_lower_cut 0  
    // intersect_leaf_zsphere t1sph   210.7622 t2sph   211.6396 sdisc     0.0000 
    // intersect_leaf_zsphere z1sph    -0.4388 z2sph     0.4386 zmax   -39.0000 zmin  -194.1000 sdisc     0.0000 
    //intersect_leaf_zsphere t1sph 210.762 t2sph 211.640 t_QCAP 172.201 t_PCAP  17.100 t1cap  17.100 t2cap 172.201  
    //intersect_leaf_zsphere  t1cap_disqualify 1 t2cap_disqualify 1 
    //intersect_leaf_zsphere valid_isect 0 t_min   0.000 t1sph 210.762 t1cap   0.000 t2cap   0.000 t2sph 211.640 t_cand   0.000 
    //]intersect_leaf_zsphere valid_isect 0 
    //]intersect_leaf typecode 103 name zsphere valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 8 node_or_leaf 1 nd_isect (    0.0000     0.0000     0.0000    -0.0000) 
    //intersect_tree  nodeIdx 9 CSG::Name   cylinder depth 3 elevation 0 
    //intersect_tree  nodeIdx 9 node_or_leaf 1 
    //intersect_node typecode 105 name cylinder 
    //[intersect_leaf typecode 105 name cylinder gtransformIdx 4 
    //]intersect_leaf typecode 105 name cylinder valid_isect 1 isect (    0.0000     0.0000    -1.0000   171.0886)   
    //intersect_tree  nodeIdx 9 node_or_leaf 1 nd_isect (    0.0000     0.0000    -1.0000   171.0886) 

    ## first rogue intersect is with nodeIdx:9 the thinner cylinder hz 0.15    nmskTailOuterITube zrange 0.15 -0.15  : 0.30


Add more debug, interestingly c is exactly zero. I thought that was radial cut, but the ray is clearly outside the radius ?::

    //intersect_node typecode 105 name cylinder 
    //[intersect_leaf typecode 105 name cylinder gtransformIdx 4 
    //[intersect_leaf_cylinder radius   264.0000 z1    -0.1500 z2     0.1500 sizeZ     0.3000 
    //intersect_leaf_cylinder : axial ray endcap handling, a  8.345e-07 c(dd*k - md*md)          0 dd       0.09 k  2.955e+04 md     -51.57  
    //]intersect_leaf typecode 105 name cylinder valid_isect 1 isect (    0.0000     0.0000    -1.0000   171.0886)   
    //intersect_tree  nodeIdx 9 node_or_leaf 1 nd_isect (    0.0000     0.0000    -1.0000   171.0886) 


    In [2]: 0.3*0.3
    Out[2]: 0.09

    In [3]: md=-51.57

    In [4]: md*md                                                                                                                                                                   
    Out[4]: 2659.4649

    In [5]: 2.955e+04                                                                                                                                                               
    Out[5]: 29550.0

    In [6]: 2.955e+04*0.09                                                                                                                                                          
    Out[6]: 2659.5



    //intersect_tree  nodeIdx 4 CSG::Name      union depth 2 elevation 1 
    //intersect_tree  nodeIdx 4 node_or_leaf 0 
    //   4 : stack peeking : left 0 right 1 (stackIdx)            union  l: Miss     0.0000    r:Enter   171.0886     leftIsCloser 1 -> RETURN_B 
    //intersect_tree  nodeIdx 10 CSG::Name       zero depth 3 elevation 0 
    //intersect_tree  nodeIdx 11 CSG::Name       zero depth 3 elevation 0 
    //intersect_tree  nodeIdx 5 CSG::Name   cylinder depth 2 elevation 1 
    //intersect_tree  nodeIdx 5 node_or_leaf 1 
    //intersect_node typecode 105 name cylinder 
    //[intersect_leaf typecode 105 name cylinder gtransformIdx 1 
    //]intersect_leaf typecode 105 name cylinder valid_isect 0 isect (    0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 5 node_or_leaf 1 nd_isect (    0.0000     0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 2 CSG::Name      union depth 1 elevation 2 
    //intersect_tree  nodeIdx 2 node_or_leaf 0 
    //   2 : stack peeking : left 0 right 1 (stackIdx)            union  l:Enter   171.0886    r: Miss     0.0000     leftIsCloser 0 -> RETURN_A 
    //intersect_tree  nodeIdx 12 CSG::Name    zsphere depth 3 elevation 0 
    //intersect_tree  nodeIdx 12 node_or_leaf 1 
    //intersect_node typecode 103 name zsphere 
    //[intersect_leaf typecode 103 name zsphere gtransformIdx 5 
    //[intersect_leaf_zsphere radius   186.0000 b  -210.7711 c 46801.4688 
    // intersect_leaf_zsphere radius   186.0000 zmax   -39.0000 zmin  -186.1000  with_upper_cut 1 with_lower_cut 0  
    // intersect_leaf_zsphere t1sph   210.7720 t2sph   222.0488 sdisc     0.0000 
    // intersect_leaf_zsphere z1sph    -0.4290 z2sph    10.8478 zmax   -39.0000 zmin  -186.1000 sdisc     0.0000 
    //intersect_leaf_zsphere t1sph 210.772 t2sph 222.049 t_QCAP 172.201 t_PCAP  25.100 t1cap  25.100 t2cap 172.201  
    //intersect_leaf_zsphere  t1cap_disqualify 1 t2cap_disqualify 1 
    //intersect_leaf_zsphere valid_isect 0 t_min   0.000 t1sph 210.772 t1cap   0.000 t2cap   0.000 t2sph 222.049 t_cand   0.000 
    //]intersect_leaf_zsphere valid_isect 0 
    //]intersect_leaf typecode 103 name zsphere valid_isect 0 isect (   -0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 12 node_or_leaf 1 nd_isect (   -0.0000     0.0000     0.0000    -0.0000) 
    //intersect_tree  nodeIdx 13 CSG::Name   cylinder depth 3 elevation 0 
    //intersect_tree  nodeIdx 13 node_or_leaf 1 
    //intersect_node typecode 105 name cylinder 
    //[intersect_leaf typecode 105 name cylinder gtransformIdx 6 
    //]intersect_leaf typecode 105 name cylinder valid_isect 0 isect (   -0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 13 node_or_leaf 1 nd_isect (   -0.0000     0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 6 CSG::Name intersection depth 2 elevation 1 
    //intersect_tree  nodeIdx 6 node_or_leaf 0 
    //   6 : stack peeking : left 1 right 2 (stackIdx)     intersection  l: Exit     0.0000    r: Exit     0.0000     leftIsCloser 0 -> RETURN_B 
    //intersect_tree  nodeIdx 14 CSG::Name       zero depth 3 elevation 0 
    //intersect_tree  nodeIdx 15 CSG::Name       zero depth 3 elevation 0 
    //intersect_tree  nodeIdx 7 CSG::Name   cylinder depth 2 elevation 1 
    //intersect_tree  nodeIdx 7 node_or_leaf 1 
    //intersect_node typecode 105 name cylinder 
    //[intersect_leaf typecode 105 name cylinder gtransformIdx 2 
    //]intersect_leaf typecode 105 name cylinder valid_isect 0 isect (   -0.0000     0.0000     0.0000     0.0000)   
    //intersect_tree  nodeIdx 7 node_or_leaf 1 nd_isect (   -0.0000     0.0000     0.0000     0.0000) 
    //intersect_tree  nodeIdx 3 CSG::Name intersection depth 1 elevation 2 
    //intersect_tree  nodeIdx 3 node_or_leaf 0 
    //   3 : stack peeking : left 1 right 2 (stackIdx)     intersection  l: Exit     0.0000    r: Exit     0.0000     leftIsCloser 0 -> RETURN_B 
    //intersect_tree  nodeIdx 1 CSG::Name intersection depth 0 elevation 3 
    //intersect_tree  nodeIdx 1 node_or_leaf 0 
    //   1 : stack peeking : left 0 right 1 (stackIdx)     intersection  l:Enter   171.0886    r: Exit     0.0000     leftIsCloser 1 -> RETURN_A 
    2022-09-12 14:51:30.112 INFO  [4293206] [CSGSimtrace::simtrace_selection@87]  num_selection 1 num_intersect 1
    2022-09-12 14:51:30.112 INFO  [4293206] [CSGSimtrace::saveEvent@97] 
    2022-09-12 14:51:30.112 INFO  [4293206] [CSGSimtrace::saveEvent@101]  outdir /tmp/blyth/opticks/nmskSolidMaskTail__U1/CSGSimtraceTest/ALL num_selection 1 selection_simtrace.sstr (1, 4, 4, )
    Fold : symbol s base /tmp/blyth/opticks/nmskSolidMaskTail__U1/CSGSimtraceTest/ALL 
    xlim:[-422.4  422.4] ylim:[-237.6  237.6] FOCUS:[0. 0. 0.] 
    INFO:opticks.ana.pvplt:mpplt_simtrace_selection_line sts
    array([[[ 0.000e+00,  0.000e+00, -1.000e+00,  1.711e+02],
            [ 2.645e+02,  0.000e+00, -4.011e+01,  0.000e+00],
            [ 2.640e+02,  0.000e+00, -2.112e+02,  1.000e+00],
            [ 3.071e-03,  0.000e+00,  1.000e+00, -1.701e+38]]], dtype=float32)

    INFO:opticks.ana.pvplt:MPPLT_SIMTRACE_SELECTION_LINE o2i,o2i_XDIST,nrm10 cfg ['o2i', 'o2i_XDIST', 'nrm10'] 
    INFO:opticks.ana.pvplt: jj [-1] 

    In [1]:                            



issue 1 : FIXED : v. thin hz < 1mm tubs mistranslated as disc not cylinder : Observe some rare spurious halo beyond the expected face of nmskTailInner.
---------------------------------------------------------------------------------------------------------------------------------------------------------
::

    c
    ./ct.sh ana

    In [11]: w = s.simtrace[:,1,0] > 260.     

    In [15]: np.where(w)
    Out[15]: (array([216852, 349933, 387116, 615829]),)

    In [17]: s.simtrace[w,:3]
    Out[17]: 
    array([[[   0.   ,    0.   ,    1.   ,  395.232],
            [ 267.011,    0.   ,  -38.   ,    0.   ],
            [-128.   ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,    1.   ,  210.675],
            [ 261.461,    0.   ,  -38.   ,    0.   ],
            [  51.2  ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,    1.   ,  162.043],
            [ 263.904,    0.   ,  -38.   ,    0.   ],
            [ 102.4  ,    0.   ,  -51.2  ,    1.   ]],

           [[   0.   ,    0.   ,   -1.   ,  149.407],
            [ 260.668,    0.   ,  -39.3  ,    0.   ],
            [ 409.6  ,    0.   ,  -51.2  ,    1.   ]]], dtype=float32)


Problem intersect ray directions are close to, but not quite horizontal:: 

    In [19]: s.simtrace[w,3,:3]
    Out[19]: 
    array([[ 0.999,  0.   ,  0.033],
           [ 0.998,  0.   ,  0.063],
           [ 0.997,  0.   ,  0.081],
           [-0.997,  0.   ,  0.08 ]], dtype=float32)


Using simtrace selection to show the intersects leading to unexpected intersects.

CSG/tests/CSGSimtraceTest.py::

     58     if not s is None:
     59         sts = s.simtrace[s.simtrace[:,1,0] > 257.]
     60     else:
     61         sts = None
     62     pass
     63     if not sts is None:
     64         mpplt_simtrace_selection_line(ax, sts, axes=fr.axes, linewidths=2)
     65     pass


Seems to show the spurious are caused by missing intersects with the thin edge of 
the tubs nmskTailInnerITube.

AHHA, the translation uses disc when it should be using tubs::

    gc
    ./mtranslate.sh  

    2022-09-11 15:24:19.032 INFO  [3749623] [CSGGeometry::init_selection@174]  no SXYZ or SXYZW selection 
    2022-09-11 15:24:19.032 INFO  [3749623] [CSGDraw::draw@57] GeoChain::convertSolid converted CSGNode tree axis Z
    2022-09-11 15:24:19.032 INFO  [3749623] [CSGDraw::draw@58]  type 113 CSG::Name(type) disc IsTree 0 width 1 height 1

     di                           
    0                             
                                  
::

    022-09-11 15:24:19.027 INFO  [3749623] [X4SolidTree::Draw@61] ]
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_@1108] [ 0 soname nmskTail_inner_PartI_Tube lvname nmskTail_inner_PartI_Tube
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::Banner@86]  lvIdx     0 soIdx     0 soname nmskTail_inner_PartI_Tube lvname nmskTail_inner_PartI_Tube
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@109] [ convert nmskTail_inner_PartI_Tube lvIdx 0
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::init@185] [ X4SolidBase identifier a entityType                   25 entityName               G4Tubs name                nmskTail_inner_PartI_Tube root 0x0
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::convertTubs@1050]  has_deltaPhi 0 pick_disc 1 deltaPhi_segment_enabled 1 is_x4tubsnudgeskip 0 do_nudge_inner 1
    2022-09-11 15:24:19.027 INFO  [3749623] [X4Solid::init@221] ]
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@127]  hint_external_bbox  0 expect_external_bbox 0 set_external_bbox  0
    2022-09-11 15:24:19.027 INFO  [3749623] [*X4Solid::Convert@138] ]
    2022-09-11 15:24:19.028 INFO  [3749623] [NTreeProcess<nnode>::init@159]  NOT WITH_CHOPPER 
    2022-09-11 15:24:19.028 INFO  [3749623] [NTreeProcess<nnode>::init@165]  want_to_balance NO y when height0 exceeds MaxHeight0  balancer.height0 0 MaxHeight0 3
    2022-09-11 15:24:19.028 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_FromRawNode@1156]  after NTreeProcess:::Process 
    2022-09-11 15:24:19.028 INFO  [3749623] [*X4PhysicalVolume::ConvertSolid_FromRawNode@1165] [ before NCSG::Adopt 
    2022-09-11 15:24:19.028 INFO  [3749623] [*NCSG::Adopt@165]  [  soIdx 0 lvIdx 0
    2022-09-11 15:24:19.028 INFO  [3749623] [*NCSG::MakeNudger@276]  treeidx 0 nudgeskip 0




* nmskTailOuterITube zrange 0.15 -0.15  : 0.30
* nmskTailOuter lip zrange -39.00 -39.30

* nmskTailInnerITube  0.65 -0.65  : 1.30
* nmskTailInner lip zrange  -38.00 -39.30

* both the lips have hz less than 1mm so they are getting translated as disc 
* THIS EXPLAINS THE LACK OF EDGE INTERSECTS 


::

    0986 const float X4Solid::hz_disc_cylinder_cut = 1.f ; // 1mm 


    1022 void X4Solid::convertTubs()
    1023 { 
    1024     const G4Tubs* const solid = static_cast<const G4Tubs*>(m_solid);
    1025     assert(solid);
    1026     //LOG(info) << "\n" << *solid ; 
    1027 
    1028     // better to stay double until there is a need to narrow to float for storage or GPU 
    1029     double hz = solid->GetZHalfLength()/mm ;
    1030     double  z = hz*2.0 ;   // <-- this full-length z is what GDML stores
    1031 
    1032     double startPhi = solid->GetStartPhiAngle()/degree ;
    1033     double deltaPhi = solid->GetDeltaPhiAngle()/degree ;
    1034     double rmax = solid->GetOuterRadius()/mm ;
    1035 
    1036     bool pick_disc = hz < hz_disc_cylinder_cut ;
    1037 
    1038     bool is_x4tubsnudgeskip = isX4TubsNudgeSkip()  ;
    1039     bool do_nudge_inner = is_x4tubsnudgeskip ? false : true ;   // --x4tubsnudgeskip 0,1,2  # lvIdx of the tree 
    1040 
    1041     nnode* tube = pick_disc ? convertTubs_disc() : convertTubs_cylinder(do_nudge_inner) ;
    1042 
    1043     bool deltaPhi_segment_enabled = true ;
    1044     bool has_deltaPhi = deltaPhi < 360. ;
    1045 







