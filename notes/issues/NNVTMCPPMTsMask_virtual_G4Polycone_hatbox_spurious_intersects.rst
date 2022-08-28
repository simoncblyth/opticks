NNVTMCPPMTsMask_virtual_G4Polycone_hatbox_spurious_intersects
================================================================

Prev:

* :doc:`gxt_MOI_shakedown`
* :doc:`gxt_running_with_single_GeoChain_translated_solid_fails_in_QSim`


TODO : unfixed PMT mask cutting across the PMT is apparent
-------------------------------------------------------------

This was fixed previously in j, 
now that the integration SVN commits are done can 
bring over the fix from j into SVN. But do it in a way to 
be easily switchable. 


"gxt.sh ana"  view 
---------------------

::

    gx

    MASK=t NOPVGRID=1 ZZ=0,1 ./gxt.sh ana

    GEOM=nmskSolidMaskVirtual MASK=t NOPVGRID=1 ZZ=0,1 ./gxt.sh ana
    
    GEOM=J003 MASK=t NOPVGRID=1 ZZ=0,1 ./gxt.sh ana


Use "geom_" to change the GEOM default::

     30 #geom=BoxOfScintillator
     31 
     32 #geom=RaindropRockAirWater
     33 #geom=RaindropRockAirWaterSD
     34 #geom=RaindropRockAirWaterSmall
     35 
     36 #geom=hama_body_log
     37 #geom=J001
     38 #geom=J003
     39 geom=nmskSolidMaskVirtual
     40 
     41 export GEOM=${GEOM:-$geom}
     42 
     

Use ISEL selection to make the simtrace easier to follow
------------------------------------------------------------

::

    GEOM=J003 NOPVGRID=1 ./gxt.sh ana
    GEOM=J003 MASK=t NOPVGRID=1 ISEL=0,2,3,4,5,8 ZZ=0,1 ./gxt.sh ana



      0 :  3094 : 141466 :                  red :                                                          NNVTMCPPMTsMask_virtual         
      2 :  3100 :  64642 :                 blue :                                          NNVTMCPPMT_PMT_20inch_inner2_solid_head 
      3 :  2325 :  62777 :                 cyan :                                                                   sReflectorInCD 
      4 :  3099 :  50721 :              magenta :                                          NNVTMCPPMT_PMT_20inch_inner1_solid_head 
      5 :  3096 :  34363 :               yellow :                                                                   NNVTMCPPMTTail 
      8 :  3095 :  25762 :           blueviolet :                                                                  NNVTMCPPMTsMask 


    positions_pvplt feat.name pid 
      4 :  3096 :  32228 :              magenta :                                                                   NNVTMCPPMTTail 
      6 :  3095 :  24219 :                 pink :                                                                  NNVTMCPPMTsMask 
    pvplt_parallel_lines
    gslim {0: array([-557.22 ,  557.221], dtype=float32), 1: array([-0.   ,  0.001], dtype=float32), 2: array([-313.438,  313.438], dtype=float32)} 
    aa    {0: [], 1: [], 2: []} 
    axes  (0, 2) 
    look  [0.0, 0.0, 0.0] 
    frame.pv_compose look:[0. 0. 0.] eye: [    0.    -2089.573     0.   ] up:[0. 0. 1.]  PARA:False RESET:0 ZOOM:1.0  
    /Users/blyth/.opticks/ntds3/G4CXOpticks/G4CXSimtraceTest/ALL/figs/positions_pvplt_pid.png

    In [1]:                                                                                                                                                                                                   
    epsilon:g4cx blyth$ 
    epsilon:g4cx blyth$ GEOM=J003 NOPVGRID=1 ISEL=4,6 ./gxt.sh ana


From jps/PMTSim "NNVTMaskManager::getSolid" the relevant mask names are::

    SolidMaskVirtual
    SolidMask
    SolidMaskTail



Use morton codes to select spurious isolated intersects for nmskSolidMask
-----------------------------------------------------------------------------

* https://blog.claude.nl/tech/timing-morton-code-on-python-on-apple-silicon/

::

    161     t_pos = SimtracePositions(t.simtrace, gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )
    162     print(t_pos)
    163 
    164     if SPURIOUS:
    165         u_kpos, c_kpos, i_kpos, t_spos = spurious_2d_outliers( t.sframe.bbox, t_pos.upos )
    166     else:
    167         t_spos = None
    168     pass


::

    GEOM=nmskSolidMask SPURIOUS=1 MASK=t ZZ=0.099,0.101 XX=37.042,37.044 ./gxt.sh ana

    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos [128130] 
    INFO:opticks.ana.pvplt:upos[i_kpos] [[37.043  0.     0.1    1.   ]] 


    GEOM=nmskSolidMask ./gxt.sh ana



HMM: what would be useful is to rerun the index with spurious intersect using the simtrace origin and direction
with both the CPU and GPU intersects 


::

   CSG/tests/CSGQueryTest.sh
   CSG/tests/CSGQueryTest.cc

Did this in CSG/SimtraceRerunTest.sh 


HMM so need to get the simtrace index, at moment have upos index::

    In [2]: t_pos.upos.shape
    Out[2]: (222743, 4)

    In [3]: t.simtrace.shape
    Out[3]: (627000, 4, 4)

As t_pos holds the mask can workout the origin simtrace index::

    In [5]: t_pos.mask.shape
    Out[5]: (627000,)

    In [7]: np.where(t_pos.mask)[0]
    Out[7]: array([     7,     18,     38,     68,     83, ..., 626961, 626963, 626976, 626982, 626983])

    In [8]: np.where(t_pos.mask)[0].shape
    Out[8]: (222743,)

    In [9]: wpos = np.where(t_pos.mask)[0] ; wpos
    Out[9]: array([     7,     18,     38,     68,     83, ..., 626961, 626963, 626976, 626982, 626983])

    In [16]:  j_kpos = wpos[i_kpos][0] ; j_kpos
    Out[16]: 348547

    In [17]: jp = t.simtrace[j_kpos] ; jp 
    Out[17]: 
    array([[ -0.   ,  -0.   ,  -1.   ,  80.85 ],
           [ 37.043,   0.   ,   0.1  ,   0.   ],
           [ 52.8  ,   0.   , -79.2  ,   0.   ],
           [ -0.195,   0.   ,   0.981,   0.   ]], dtype=float32)


    In [20]: jp[3,:3]
    Out[20]: array([-0.195,  0.   ,  0.981], dtype=float32)

    In [21]: jp[2,:3]
    Out[21]: array([ 52.8,   0. , -79.2], dtype=float32)

    In [22]: jp[2,:3] + jp[0,3]*jp[3,:3]      ## origin + dist*direction  at intersect 
    Out[22]: array([37.043,  0.   ,  0.1  ], dtype=float32)


Automate the back mapping::

    In [6]: t_pos.upos2simtrace[i_kpos]
    Out[6]: array([176995, 153452, 459970])

    In [7]: j_kpos = t_pos.upos2simtrace[i_kpos]

    In [8]: simtrace[j_kpos]
    Out[8]: 
    array([[[  -0.   ,   -0.   ,   -1.   ,  125.124],
            [-117.841,    0.   ,    0.1  ,   40.1  ],
            [-184.8  ,    0.   , -105.6  ,    0.   ],
            [   0.535,    0.   ,    0.845,    0.   ]],

           [[  -0.   ,   -0.   ,   -1.   ,  239.297],
            [ -98.882,    0.   ,    0.1  ,   40.1  ],
            [-211.2  ,    0.   , -211.2  ,    0.   ],
            [   0.469,    0.   ,    0.883,    0.   ]],

           [[  -0.   ,   -0.   ,   -1.   ,  185.968],
            [ 113.929,    0.   ,    0.1  ,   40.1  ],
            [ 211.2  ,    0.   , -158.4  ,    0.   ],
            [  -0.523,    0.   ,    0.852,    0.   ]]], dtype=float32)


::

    INFO:opticks.ana.pvplt:SPURIOUS envvars switches on morton enabled spurious_2d_outliers 
    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos [ 43865  34010 181781] 
    INFO:opticks.ana.pvplt:upos[i_kpos] [[-117.841    0.       0.1      1.   ]
     [ -98.882    0.       0.1      1.   ]
     [ 113.929    0.       0.1      1.   ]] 
    INFO:opticks.ana.pvplt:j_kpos = t_pos.upos2simtrace[i_kpos]
    [176995 153452 459970]
    INFO:opticks.ana.pvplt:simtrace[j_kpos]
    [[[  -0.      -0.      -1.     125.124]
      [-117.841    0.       0.1     40.1  ]
      [-184.8      0.    -105.6      0.   ]
      [   0.535    0.       0.845    0.   ]]

     [[  -0.      -0.      -1.     239.297]
      [ -98.882    0.       0.1     40.1  ]
      [-211.2      0.    -211.2      0.   ]
      [   0.469    0.       0.883    0.   ]]

     [[  -0.      -0.      -1.     185.968]
      [ 113.929    0.       0.1     40.1  ]
      [ 211.2      0.    -158.4      0.   ]
      [  -0.523    0.       0.852    0.   ]]]


     SELECTION=176995,153452,459970 ./SimtraceRerunTest.sh 




CPU rerun using CSG/SimtraceRerunTest.sh does not have that particular spurious intersect::

    In [31]: t.simtrace[348547]
    Out[31]: 
    array([[ -0.   ,  -0.   ,  -1.   ,  80.85 ],
           [ 37.043,   0.   ,   0.1  ,   0.   ],
           [ 52.8  ,   0.   , -79.2  ,   0.   ],
           [ -0.195,   0.   ,   0.981,   0.   ]], dtype=float32)

    In [32]: t.simtrace_rerun[348547]
    Out[32]: 
    array([[ -0.   ,  -0.   ,  -1.   , 270.385],
           [  0.105,   0.   , 186.   ,   0.   ],
           [ 52.8  ,   0.   , -79.2  ,   0.   ],
           [ -0.195,   0.   ,   0.981,   0.   ]], dtype=float32)


But visualizing the simtrace_rerun, shows it has three suprious intersects on that same z=0.1 line::

    ZZ=0.1 RERUN=1 ./gxt.sh ana


Find their indices using morton magic::

     GEOM=nmskSolidMask MASK=t RERUN=1 SPURIOUS=1 ./gxt.sh ana

::

    INFO:opticks.ana.pvplt:RERUN envvar switched on use of simtrace_rerun from CSG/SimtraceRerunTest.sh 
    INFO:opticks.ana.simtrace_positions:apply_t_mask
    SimtracePositions
    t_pos.simtrace (222742, 4, 4) 
    t_pos.isect (627000, 4) 
    t_pos.gpos (627000, 4) 
    t_pos.lpos (627000, 4) 
    INFO:opticks.ana.pvplt:SPURIOUS envvars switches on morton enabled spurious_2d_outliers 
    INFO:opticks.ana.pvplt:spurious_2d_outliers
    INFO:opticks.ana.pvplt:i_kpos [ 43865  34010 181781] 
    INFO:opticks.ana.pvplt:upos[i_kpos] [
     [-117.841    0.       0.1      1.   ]
     [ -98.882    0.       0.1      1.   ]
     [ 113.929    0.       0.1      1.   ]] 



Rerun the three spurious::

    epsilon:CSG blyth$ SELECTION=176995,153452,459970 ./SimtraceRerunTest.sh 
                       BASH_SOURCE : ./../bin/GEOM_.sh 
                               gp_ : nmskSolidMask_GDMLPath 
                                gp :  
                               cg_ : nmskSolidMask_CFBaseFromGEOM 
                                cg : /tmp/blyth/opticks/GeoChain/nmskSolidMask 
                       TMP_GEOMDIR : /tmp/blyth/opticks/nmskSolidMask 
                           GEOMDIR : /tmp/blyth/opticks/GeoChain/nmskSolidMask 
    ...
    2022-08-27 16:34:27.512 INFO  [39352531] [CSGQuery::init@65]  sopr 0:0 solidIdx 0 primIdxRel 0
    2022-08-27 16:34:27.513 INFO  [39352531] [SimtraceRerunTest::init@69]  fd.geom (null)
    2022-08-27 16:34:27.513 INFO  [39352531] [CSGDraw::draw@30] SimtraceRerunTest axis Z
    2022-08-27 16:34:27.513 INFO  [39352531] [CSGDraw::draw@32]  type 2 CSG::Name(type) intersection IsTree 1 width 7 height 2

                                   in                                                         
                                  1                                                           
                                     0.00                                                     
                                    -0.00                                                     
                                                                                              
               un                                      in                                     
              2                                       3                                       
                 0.00                                    0.00                                 
                -0.00                                   -0.00                                 
                                                                                              
     zs                  cy                 !zs                 !cy                           
    4                   5                   6                   7                             
     194.00                0.10              186.00                0.10                       
     -39.00              -38.90              -40.00              -39.90                       
                                                                                              
                                                                                              
                                                                                              
    

::

     64     G4Ellipsoid(const G4String& pName,
     65                       G4double  pxSemiAxis,
     66                       G4double  pySemiAxis,
     67                       G4double  pzSemiAxis,
     68                       G4double  pzBottomCut=0,
     69                       G4double  pzTopCut=0);



jps/tests/GetValuesTest:: 

    PMTSim::getValues name_ [nmskSolidMask] name [SolidMask] mgr Y NAME_OFFSET 0 vv (15, )
     name nmskSolidMask vv (15, )
    NP::descValues num_val 15

      0 v   264.0000 k  SolidMask.Top_out.pxySemiAxis.mask_radiu_out
      1 v   194.0000 k  SolidMask.Top_out.pzSemiAxis.htop_out
      2 v   -39.0000 k  SolidMask.Top_out.pzBottomCut.-height_out
      3 v   194.0000 k  SolidMask.Top_out.pzTopCut.htop_out

      4 v    19.5000 k  SolidMask.Bottom_out.hz.height_out/2
      5 v   -19.4000 k  SolidMask.Mask_out.zoffset.-height_out/2+gap

      6 v   256.0000 k  SolidMask.Top_in.pxySemiAxis.mask_radiu_in
      7 v   186.0000 k  SolidMask.Top_in.pzSemiAxis.htop_in
      8 v   -40.0000 k  SolidMask.Top_in.pzBottomCut.-(height_in+uncoincide_z)
      9 v   186.0000 k  SolidMask.Top_in.pzTopCut.htop_in

     10 v    20.0000 k  SolidMask.Bottom_in.hz.height_in/2 + uncoincide_z/2

     11 v   -19.9000 k  SolidMask.Mask_in.zoffset.-height_in/2 + gap - uncoincide_z/2
     12 v   -19.5000 k  SolidMask.Mask_in.zoffset.-height_in/2
     13 v     0.1000 k  SolidMask.Mask_in.zoffset.gap
     14 v    -0.5000 k  SolidMask.Mask_in.zoffset.-uncoincide_z/2


::

   ZZ=194 ./gxt.sh ana


   ELLIPSE0=264,194,0,0,0.1,-39,0 ZZ=194,-39 ./gxt.sh ana
   ELLIPSE0=264,194,0,0,0.1,-39,0 ZZ=194,-39 RECTANGLE0=264,19.5,0,0,0.3,-19.4 ./gxt.sh ana

   ELLIPSE0=264,194,0,0,0.1,-39,0 ZZ=194,-39 RECTANGLE0=264,19.5,0,0,0.3,-19.4 RECTANGLE1=256,20,0,0,0.3,-19.9   ./gxt.sh ana




   ELLIPSE1=256,186,0,0,0.1,-40,0 ZZ=186,-40 ./gxt.sh ana
   ELLIPSE1=256,186,0,0,0.1,-40,0 ZZ=186,-40 RECTANGLE1=256,20,0,0,0.3,-19.9 ./gxt.sh ana




                                                                                          
                                                                                              
                                                                                              
     idx  176995 code 3
                            isect0 HIT
                        q0 norm t (    0.0002    0.0000   -1.0000  345.1852)
                       q1 ipos sd (   -0.0780    0.0000  186.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -184.8000    0.0000 -105.6000    0.0000)
                  q3 ray_dir gsid (    0.5351    0.0000    0.8448 C4U (     0    0    0    0 ) )

                            isect1 HIT
                        q0 norm t (   -0.0000   -0.0000   -1.0000  125.1237)
                       q1 ipos sd ( -117.8414    0.0000    0.1000   40.1000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -184.8000    0.0000 -105.6000    0.0000)
                  q3 ray_dir gsid (    0.5351    0.0000    0.8448 C4U (     0    0    0    0 ) )

     idx  153452 code 3
                            isect0 HIT
                        q0 norm t (    0.0002    0.0000   -1.0000  449.8282)
                       q1 ipos sd (   -0.0659    0.0000  186.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -211.2000    0.0000 -211.2000    0.0000)
                  q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )

                            isect1 HIT
                        q0 norm t (   -0.0000   -0.0000   -1.0000  239.2969)
                       q1 ipos sd (  -98.8822    0.0000    0.1000   40.1000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min ( -211.2000    0.0000 -211.2000    0.0000)
                  q3 ray_dir gsid (    0.4694    0.0000    0.8830 C4U (     0    0    0    0 ) )

     idx  459970 code 3
                            isect0 HIT
                        q0 norm t (    0.0004    0.0000   -1.0000  404.0836)
                       q1 ipos sd (   -0.1580    0.0000  186.0000    0.0000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  211.2000    0.0000 -158.4000    0.0000)
                  q3 ray_dir gsid (   -0.5231    0.0000    0.8523 C4U (     0    0    0    0 ) )

                            isect1 HIT
                        q0 norm t (   -0.0000   -0.0000   -1.0000  185.9677)
                       q1 ipos sd (  113.9287    0.0000    0.1000   40.1000)- sd < SD_CUT :    -0.0010
                 q2 ray_ori t_min (  211.2000    0.0000 -158.4000    0.0000)
                  q3 ray_dir gsid (   -0.5231    0.0000    0.8523 C4U (     0    0    0    0 ) )

    2022-08-27 16:34:27.514 INFO  [39352531] [main@148] t.desc SimtraceRerunTest::desc
     fd Y
     path0 /tmp/blyth/opticks/GeoChain/nmskSolidMask/G4CXSimtraceTest/ALL/simtrace.npy
     path1 /tmp/blyth/opticks/GeoChain/nmskSolidMask/G4CXSimtraceTest/ALL/simtrace_rerun.npy
     simtrace0 (627000, 4, 4, )
     simtrace1 (627000, 4, 4, )
     selection Y selection.size 3
     code_count[0] 0
     code_count[1] 0
     code_count[2] 0
     code_count[3] 3
     code_count[4] 3












::

    269 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    270 {
    271     unsigned idx = launch_idx.x ;  // aka photon_id
    272     sevent* evt  = params.evt ;
    273     if (idx >= evt->num_simtrace) return;
    274 
    275     unsigned genstep_id = evt->seed[idx] ;
    276     if(idx == 0) printf("//OptiX7Test.cu:simtrace idx %d genstep_id %d \n", idx, genstep_id );
    277 
    278     const quad6& gs     = evt->genstep[genstep_id] ;
    279 
    280     qsim* sim = params.sim ;
    281     curandState rng = sim->rngstate[idx] ;
    282 
    283     quad4 p ;
    284     sim->generate_photon_simtrace(p, rng, gs, idx, genstep_id );
    285 
    286     const float3& pos = (const float3&)p.q0.f  ;
    287     const float3& mom = (const float3&)p.q1.f ;
    288 
    289     trace(
    290         params.handle,
    291         pos,
    292         mom,
    293         params.tmin,
    294         params.tmax,
    295         prd
    296     );
    297 
    298     evt->add_simtrace( idx, p, prd, params.tmin );
    299 
    300 }






How to investigate spurious intersects
----------------------------------------

* add G4VSolid implementation to U4VolumeMaker (or PMTSim) 
  and test in isolation  using GeoChain

* try getting the csg intersect machinery on CPU to give the same thing 

* check with Geant4 X4SolidIntersect  


Investigate Issue 3 with GeoChain
-------------------------------------

geom::

    nmskSolidMaskVirtual_XZ


gc::

   ./translate.sh   



Issue 3 : Note some slop intersects from NNVTMCPPMTsMask_virtual hatbox G4Polycone
--------------------------------------------------------------------------------------

* some on union coincidence plane between polycone and cylinder 

  * actually whole shape is a single G4Polycone with 4 planes, 
    it seems the anti-coincidence is not working possibly 
    due to equal radii 

  * this is an overcomplicated and expensive way to implement 
    the cylinder part of the hatbox : using 3 polycone planes 

   * HMM the Opticks G4Polycone translation could notice the 
     equal radii and hence simplify the modelling in the translation


   * DONE: get the shape from PMTSim nmsk into GeoChain
     
     * while doing this can think about more direct shape conversion 

* also some unexpected ones mid-cylinder 

  * using ZZ=0,1 shows that they are on the z=1mm plane 
  * which is unexpected as the implementation makes it look like the 
    G4Polycone plane is at 0 ?  Did the anti-coincicence kick in wrong somehow ?
  * potentially changing to use 3 planes, not 4, could avoid the issue 
    and simplify the shape

* the upper plane joint has more of a problem 
  and seems no easy way to anticoincide because growing either shape into 
  the other would change the shape 

  * changing shape a little with the radius of the upper cone starting slightly
    less than the cylinder radius would allow the cone to extend down slightly 
    overlapping into the cylinder and avoid the coincident plane


::

    MASK=t NOPVGRID=1 ZZ=0,1 ./gxt.sh ana





::

    244 void
    245 NNVTMaskManager::makeMaskOutLogical() {
    ...
    268     // BELOW is using 4 zplanes
    269     G4double zPlane[] = {
    270                         -height_virtual,
    271                         0, // at equator
    272                         htop_out/2, // at half H_front
    273                         htop_out + MAGIC_virtual_thickness
    274                         };
    275     G4double rInner[] = {0.,
    276                          0., // at equator
    277                          0., // at half H_front
    278                          0.};
    279     G4double rOuter[] = {mask_radiu_virtual,
    280                          mask_radiu_virtual, // at equator
    281                          mask_radiu_virtual, // at half H_front
    282                          mask_radiu_virtual/2}; // reduce the front R
    283 
    284 
    285     G4VSolid* SolidMaskVirtual = new G4Polycone(
    286                 objName()+"sMask_virtual",
    287                                 0,
    288                                 360*deg,
    289                                 // 2,
    290                                 4,
    291                                 zPlane,
    292                                 rInner,
    293                                 rOuter
    294                                 );






::

    positions_pvplt feat.name pid 
      0 :  3094 : 106024 :                  red :                                                          NNVTMCPPMTsMask_virtual 

::

   ZZ=0,1 ISEL=0 ./gxt.sh ana






