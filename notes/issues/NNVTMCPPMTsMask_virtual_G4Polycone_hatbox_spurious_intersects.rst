NNVTMCPPMTsMask_virtual_G4Polycone_hatbox_spurious_intersects
================================================================

Prev:

* :doc:`gxt_MOI_shakedown`
* :doc:`gxt_running_with_single_GeoChain_translated_solid_fails_in_QSim`


TODO : unfixed PMT mask cutting across the PMT is apparent
-------------------------------------------------------------

This was fixed previously in j, 
now that the integration SVN commits are done can 
bring over the fix from j into SVN. 


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


HMM: what would be useful is to rerun the index with spurious intersect using the simtrace origin and direction
with both the CPU and GPU intersects 

::

   CSG/tests/CSGQueryTest.sh
   CSG/tests/CSGQueryTest.cc

Start::

   CSG/CSGFoundrySimtraceRerunTest.sh
   CSG/tests/CSGFoundrySimtraceRerunTest.cc 



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






