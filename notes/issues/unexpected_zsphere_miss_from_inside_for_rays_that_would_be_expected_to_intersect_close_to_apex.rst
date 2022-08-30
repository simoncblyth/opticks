unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex
==================================================================================================

* fixed by changing "<" to "<="  in intersect_leaf_zsphere


* next :doc:`nmskSolidMaskTail`


Following apex fix nmskSolidMask note an apparently harmless irregularity on bottom edge : sligthly lower nearer inside
--------------------------------------------------------------------------------------------------------------------------


This selection rerun shows the apex bug : fixed by upper placeholder zcut safety
-----------------------------------------------------------------------------------

c::

    ./nmskSolidMask.sh

    selection=153452
    GEOM=nmskSolidMask SELECTION=$selection ./CSGSimtraceRerunTest.sh  


isect1 is the rerun::

             isect0 HIT
                    q0 norm t (    0.0002    0.0000   -1.0000  449.8282)
                   q1 ipos sd (   -0.0659    0.0000  186.0000    0.0000)- sd < SD_CUT :    -0.0010
             q2 ray_ori t_min ( -211.2000    0.0000 -211.2000)
              q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )

             isect1 HIT
                    q0 norm t (   -0.0000   -0.0000   -1.0000  239.2969)
                   q1 ipos sd (  -98.8822    0.0000    0.1000   40.1000)- sd < SD_CUT :    -0.0010
             q2 ray_ori t_min ( -211.2000    0.0000 -211.2000)
              q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )

gx::

    SELECTION=1 ./gxt.sh 

::

    .    const float2 zdelta = make_float2(q1.f);
    -    const float zmax = center.z + zdelta.y ; 
    +    const float zmax = center.z + zdelta.y + 0.1f ;  // artificial increase zmax to test apex bug 
         const float zmin = center.z + zdelta.x ;    


After the artificial zmax increase get the expected near apex hit::

             isect0 HIT
                    q0 norm t (    0.0002    0.0000   -1.0000  449.8282)
                   q1 ipos sd (   -0.0659    0.0000  186.0000    0.0000)- sd < SD_CUT :    -0.0010
             q2 ray_ori t_min ( -211.2000    0.0000 -211.2000)
              q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )

             isect1 HIT
                    q0 norm t (    0.0002   -0.0000   -1.0000  449.8283)
                   q1 ipos sd (   -0.0659    0.0000  186.0000   -0.0000)- sd < SD_CUT :    -0.0010
             q2 ray_ori t_min ( -211.2000    0.0000 -211.2000)
              q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )


Hmm how to formalize this, do it in translation::

    epsilon:extg4 blyth$ grep zsphere *.*
    X4Solid.cc:        cn = nzsphere::Create( x, y, z, radius, zmin, zmax ) ;
    X4Solid.cc:        cn->label = BStr::concat(m_name, "_nzsphere", NULL) ; 
    X4Solid.cc:    nzsphere 
    X4Solid.cc:                          (nnode*)nzsphere::Create( 0.f, 0.f, 0.f, cz, z1, z2 ) 
    epsilon:extg4 blyth$ 
    epsilon:extg4 blyth$ 


::

    1480 void X4Solid::convertEllipsoid()
    1481 {
    1482     const G4Ellipsoid* const solid = static_cast<const G4Ellipsoid*>(m_solid);
    1483     assert(solid);
    1484 
    1485     // G4GDMLWriteSolids::EllipsoidWrite
    1486 
    1487     double ax = solid->GetSemiAxisMax(0)/mm ;
    1488     double by = solid->GetSemiAxisMax(1)/mm ;
    1489     double cz = solid->GetSemiAxisMax(2)/mm ;
    1490 
    1491     glm::tvec3<double> scale( ax/cz, by/cz, 1.) ;   // unity scaling in z, so z-coords are unaffected  
    1492 
    1493     double zcut1 = solid->GetZBottomCut()/mm ;
    1494     double zcut2 = solid->GetZTopCut()/mm ;
    1495 
    1496     double z1 = zcut1 > -cz ? zcut1 : -cz ;
    1497     double z2 = zcut2 <  cz ? zcut2 :  cz ;
    1498     assert( z2 > z1 ) ;
    1499 
    1500     bool upper_cut = z2 < cz ;
    1501     bool lower_cut = z1 > -cz ;
    1502     bool zslice = lower_cut || upper_cut ;
    1503 
    ....
    1516 
    1517     nnode* cn = nullptr ;
    1518     if( upper_cut == false && lower_cut == false )
    1519     {   
    1520         cn =  (nnode*)nsphere::Create( 0.f, 0.f, 0.f, cz ) ;
    1521     }
    1522     else if( upper_cut == true && lower_cut == true )
    1523     {   
    1524         cn = (nnode*)nzsphere::Create( 0.f, 0.f, 0.f, cz, z1, z2 )  ;
    1525     }
    1526     else if ( upper_cut == false && lower_cut == true )   // PMT mask uses this 
    1527     {   
    1528         double z2_safe = z2 + 0.1 ;  // trying to avoid the apex bug  
    1529         cn = (nnode*)nzsphere::Create( 0.f, 0.f, 0.f, cz, z1, z2_safe )  ; 
    1530         // when there is no upper cut avoid the placeholder upper cut from ever doing anything by a safety offset
    1531         // see notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst
    1532     }
    1533     else if ( upper_cut == true && lower_cut == false )
    1534     {   
    1535         double z1_safe = z1 - 0.1 ; // also avoid analogous nadir bug 
    1536         cn = (nnode*)nzsphere::Create( 0.f, 0.f, 0.f, cz, z1_safe, z2 )  ; 
    1537         // when there is no lower cut avoid the placeholder lower cut from ever doing anything by a safety offset
    1538         // see notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst
    1539     }
    1540     
    1541     
    1542     cn->label = BStr::concat(m_name, "_ellipsoid", NULL) ;





Perhaps a related issue with nmskSolidMask for intersects close to apex
-------------------------------------------------------------------------------

* the safe upper cut when no upper cut avoids the apex issue

On GPU running has one out of 300k spurious intersect, select and plot it::

    epsilon:g4cx blyth$ MASK=t SPURIOUS=1 ./gxt.sh 


    INFO:opticks.ana.pvplt:SPURIOUS envvars switches on morton enabled spurious_2d_outliers 
    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos
    [128130]
    INFO:opticks.ana.pvplt:upos[i_kpos]
    [[37.043  0.     0.1    1.   ]]
    INFO:opticks.ana.pvplt:j_kpos = t_pos.upos2simtrace[i_kpos]
    [348547]
    INFO:opticks.ana.pvplt:simtrace[j_kpos]
    [[[ -0.     -0.     -1.     80.85 ]
      [ 37.043   0.      0.1     0.   ]
      [ 52.8     0.    -79.2     0.   ]
      [ -0.195   0.      0.981   0.   ]]]


    epsilon:g4cx blyth$ SPURIOUS=1 MASK=t XDIST=300 ZZ=186 ./gxt.sh  


Doing a simtrace CPU rerun gives two spurious intersects along same z=0.1::

    c ; ./nmskSolidMask.sh   
  
Running this after recompiling with DEBUG DEBUG_RECORD gives lots of details::

    c ; ./nmskSolidMask.sh 


XDIST extending gives expected intersect close to apex of inner zsphere::

    epsilon:g4cx blyth$ RERUN=1 SPURIOUS=1 MASK=t XDIST=500 ZZ=186 ./gxt.sh  


::

    epsilon:g4cx blyth$ GEOM=nmskSolidMask RERUN=1 MASK=t SELECTION=1 ZZ=186 ./gxt.sh 

::

     ./gxt.sh mpcap
     ./gxt.sh mppub
     PUB=message ./gxt.sh mppub






Checking nmskMaskOut revealed some rare unexpected misses from inside zsphere
------------------------------------------------------------------------------

The expected intersect that was missed is close to "apex"::

    In [2]: dir = np.array( [0.135,0.,0.991] )
    In [3]: ori = np.array( [-26.4,0,0 ] )
    In [5]: ori+dir*195.792694 
    Out[5]: array([  0.032,   0.   , 194.031])



::

    epsilon:g4cx blyth$ MASK=t SPURIOUS=1 ./gxt.sh ana


    INFO:opticks.ana.pvplt:SPURIOUS envvars switches on morton enabled spurious_2d_outliers 
    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos
    [121331 133941]
    INFO:opticks.ana.pvplt:upos[i_kpos]
    [[-26.386   0.      0.1     1.   ]
     [ -0.      0.      0.1     1.   ]]
    INFO:opticks.ana.pvplt:j_kpos = t_pos.upos2simtrace[i_kpos]
    [294748 313463]
    INFO:opticks.ana.pvplt:simtrace[j_kpos]
    [[[  0.      0.      1.      0.101]
      [-26.386   0.      0.1     0.05 ]
      [-26.4     0.      0.      0.   ]
      [  0.135   0.      0.991   0.   ]]

     [[  0.      0.      1.      0.1  ]
      [ -0.      0.      0.1     0.05 ]
      [  0.      0.      0.      0.   ]
      [ -0.      0.      1.      0.   ]]]
        


::

    MASK=t RERUN=1 SPURIOUS=1 ./gxt.sh ana

    INFO:opticks.ana.pvplt:SPURIOUS envvars switches on morton enabled spurious_2d_outliers 
    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos
    [121329]
    INFO:opticks.ana.pvplt:upos[i_kpos]
    [[-26.386   0.      0.1     1.   ]]
    INFO:opticks.ana.pvplt:j_kpos = t_pos.upos2simtrace[i_kpos]
    [294748]
    INFO:opticks.ana.pvplt:simtrace[j_kpos]
    [[[  0.      0.      1.      0.101]
      [-26.386   0.      0.1   -39.1  ]
      [-26.4     0.      0.      0.05 ]
      [  0.135   0.      0.991   0.   ]]]










It manages to miss the zs from inside it::

    //intersect_tree  nodeIdx 3 node_or_leaf 1 nd_isect (    0.0000     0.0000     1.0000     0.1009) 
    //intersect_tree  nodeIdx 1 CSG::Name      union depth 0 elevation 1 
    //intersect_tree  nodeIdx 1 node_or_leaf 0 
    //   1 : stack peeking : left 0 right 1 (stackIdx)            union  l: Miss     0.0000    r: Exit     0.1009     leftIsCloser 1 -> RETURN_B 
    //   1 CSG decision : left 0 right 1 (stackIdx)            union  l: Miss     0.0000    r: Exit     0.1009     leftIsCloser 1 -> RETURN_B 
    // intersect_tree ierr 0 csg.curr 0 
    //distance_leaf typecode 103 name zsphere complement 0 sd   -39.1000 
    //distance_leaf_cylinder sd    -0.0000 
    //distance_leaf typecode 105 name cylinder complement 0 sd    -0.0000 
     i   0 idx  294748 code 3
                            isect0 HIT
                        q0 norm t (    0.0000    0.0000    1.0000    0.1009)
                       q1 ipos sd (  -26.3864    0.0000    0.1000    0.0500)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  -26.4000    0.0000    0.0000)
                  q3 ray_dir gsid (    0.1350    0.0000    0.9908 C4U (     0    0    0    0 ) )

                            isect1 HIT
                        q0 norm t (    0.0000    0.0000    1.0000    0.1009)
                       q1 ipos sd (  -26.3864    0.0000    0.1000  -39.1000) SPURIOUS INTERSECT  sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  -26.4000    0.0000    0.0000)
                  q3 ray_dir gsid (    0.1350    0.0000    0.9908 C4U (     0    0    0    0 ) )


    2022-08-29 15:52:01.861 INFO  [40823464] [CSGSimtraceRerun::save@169]  path1 /tmp/blyth/opticks/GeoChain/nmskMaskOut/G4CXSimtraceTest/ALL/simtrace_selection.npy
    2022-08-29 15:52:01.863 INFO  [40823464] [CSGSimtraceRerun::report@176] t.desc CSGSimtraceRerun::desc
     fd Y
     fd.geom -
     CSGQuery::Label  DEBUG DEBUG_RECORD
     path0 /tmp/blyth/opticks/GeoChain/nmskMaskOut/G4CXSimtraceTest/ALL/simtrace.npy
     path1 /tmp/blyth/opticks/GeoChain/nmskMaskOut/G4CXSimtraceTest/ALL/simtrace_selection.npy
     simtrace0 (627000, 4, 4, )
     simtrace1 (1, 2, 4, 4, )
     SELECTION 294748
     selection Y selection.size 1
     with_selection 1
     code_count[0] 0
     code_count[1] 0
     code_count[2] 0
     code_count[3] 1
     code_count[4] 1

    2022-08-29 15:52:01.863 INFO  [40823464] [CSGSimtraceRerun::report@178] with : DEBUG_RECORD 
    2022-08-29 15:52:01.863 INFO  [40823464] [CSGRecord::Dump@134] CSGSimtraceRerun::report CSGRecord::record.size 1IsEnabled 0
     tloop    0 nodeIdx    1 irec          0 label                                                                                        rec union
                     r.q0.f left  (    0.0000    0.0000    0.0000   -0.0000) Miss  - - - leftIsCloser
                    r.q1.f right  (    0.0000    0.0000    1.0000    0.1009) Exit  - - -   ctrl RETURN_B
               r.q3.f tmin/t_min  (    0.0500    0.0500    0.0000    0.0000)  tmin     0.0500 t_min     0.0500 tminAdvanced     0.0000
                   r.q4.f result  (    0.0000    0.0000    1.0000    0.1009) 
    2022-08-29 15:52:01.863 INFO  [40823464] [CSGSimtraceRerun::report@181]  save CSGRecord.npy to fold /tmp/blyth/opticks/GeoChain/nmskMaskOut/G4CXSimtraceTest/ALL
    2022-08-29 15:52:01.863 INFO  [40823464] [CSGRecord::Save@247]  dir /tmp/blyth/opticks/GeoChain/nmskMaskOut/G4CXSimtraceTest/ALL num_record 1
    NP::init size 24 ebyte 4 num_char 96
    with : DEBUG 
    epsilon:CSG blyth$ 


Found the cause::

     190 LEAF_FUNC
     191 bool intersect_leaf_zsphere(float4& isect, const quad& q0, const quad& q1, const float& t_min, const float3& ray_origin, const float3& ray_direction )
     192 {
     ...
     252 #ifdef DEBUG_RECORD
     253         //std::raise(SIGINT); 
     254 #endif
     255 
     256         if(      t1sph > t_min && z1sph > zmin && z1sph <= zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
     257         else if( t1cap > t_min )                                  t_cand = t1cap ;  // t1cap qualifies -> t1cap 
     258         else if( t2cap > t_min )                                  t_cand = t2cap ;  // t2cap qualifies -> t2cap
     259         else if( t2sph > t_min && z2sph > zmin && z2sph <= zmax)   t_cand = t2sph ;  // t2sph qualifies and t2cap disabled or disqialified -> t2sph
     260 
     261 /*
     262 NB "z2sph <= zmax" changed from "z2sph < zmax" Aug 29, 2022
     263 
     264 The old inequality caused rare unexpected MISS for rays that would
     265 have been expected to intersect close to the apex of the zsphere  
     266 */
     267 


::

    (lldb) f 3
    frame #3: 0x00000001001a6e59 libCSG.dylib`intersect_leaf_zsphere(isect=0x00007ffeefbfde20, q0=0x000000010330e040, q1=0x000000010330e050, t_min=0x00007ffeefbfdadc, ray_origin=0x00007ffeefbfdaa0, ray_direction=0x00007ffeefbfda80) at csg_intersect_leaf.h:253
       250 	    {
       251 	
       252 	#ifdef DEBUG_RECORD
    -> 253 	        std::raise(SIGINT); 
       254 	#endif
       255 	
       256 	        if(      t1sph > t_min && z1sph > zmin && z1sph < zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
    (lldb) p t1sph
    (float) $0 = -191.910675
    (lldb) p t_min
    (const float) $1 = 0.0500000007
    (lldb) p z1sph
    (float) $2 = -190.153519
    (lldb) p zmin
    (const float) $3 = -39
    (lldb) p  t1sph > t_min
    (bool) $4 = false
    (lldb) p t1cap
    (float) $5 = -39.3603897
    (lldb) p t2cap
    (float) $6 = 0.0500000007
    (lldb) p t2cap > t_min
    (bool) $7 = false
    (lldb) p t2sph
    (float) $8 = 195.792694
    (lldb) p z2sph
    (float) $9 = 194
    (lldb) p zmin
    (const float) $10 = -39
    (lldb) p z2sph
    (float) $11 = 194
    (lldb) p zmax
    (const float) $12 = 194
    (lldb) p z2sph < zmax
    (bool) $13 = false
    (lldb) p z2sph <= zmax
    (bool) $14 = true
    (lldb) p ray_origin.z
    (const float) $15 = 0
    (lldb) p ray_direction.z 
    (const float) $16 = 0.990843892
    (lldb) p t2sph
    (float) $17 = 195.792694
    (lldb) p z2sph
    (float) $18 = 194


