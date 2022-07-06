gxs_ab_hama_body_log
=======================

Run and Ana
--------------

Running the sims after setting GEOM in bin/GEOM_.sh and copying that to remote
and making sure using same input photons. 

A::

    gx              
    ./gxs.sh        # workstation
    ./gxs.sh grab   # laptop

B::

    u4t
    ./U4RecorderTest.sh   # laptop
   

AB Analysis::

    gx
    vi ../bin/GEOM_.sh     # select GEOM eg "hama_body_log"
    vi ../bin/AB_FOLD.sh   # check appropriate result folders are configured  

    ./gxs_ab.sh 


First Look : hama_body_log : NB might well be shooting it in the back 
----------------------------------------------------------------------------

Lots are out of history alignment::

    In [6]: w = np.where( a.seq[:,0] != b.seq[:,0])[0] ; len(w)
    Out[6]: 2139

Quite a lot aligned too::

    In [11]: wm = np.where( a.seq[:,0] == b.seq[:,0])[0] ; len(wm)
    Out[11]: 7861


* BUT: recall that seqhis matching is not a very good indicator of alignment 
  as that will often still accidentally match even after consumption alignment is lost 

* DONE : find way to select aligned idx more stringently based on the tag/stack enumerations and/or flat 

  * converted the enum seq to a "S48" or "S64" depending on SLOTS string and compared that 


Mimic the Opticks Flags with Geant4? 
---------------------------------------

::

    1258 inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx )
    1259 {
    1260     const unsigned boundary = ctx.prd->boundary() ;
    1261     const unsigned identity = ctx.prd->identity() ;
    1262     const unsigned iindex = ctx.prd->iindex() ;
    1263     const float3* normal = ctx.prd->normal();
    1264     float cosTheta = dot(ctx.p.mom, *normal ) ;
    1265 
    1266 #ifdef DEBUG_PIDX
    1267     if( ctx.idx == base->pidx )
    1268     printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n",
    1269                  ctx.idx, bounce, cosTheta, ctx.p.mom.x, ctx.p.mom.y, ctx.p.mom.z, normal->x, normal->y, normal->z );
    1270 #endif
    1271 
    1272     ctx.p.set_prd(boundary, identity, cosTheta, iindex );
    1273 

::

    130 SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_, unsigned iindex_ )
    131 {
    132     set_boundary(boundary_);
    133     identity = identity_ ;
    134     set_orient( orient_ );
    135     iindex = iindex_ ;
    136 }

    SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } 
    // clear orient bit and then set it 
    //
    // cosTheta < 0.f : photon direction is against the normal of the geometry => 0x1 => "-"
    // cosTheta > 0.f : photon direction is with    the normal of the geometry => 0x0 => "+"  




::

    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@82]    0 bnd Rock///Rock
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@82]    1 bnd Rock//air_rock_bs/Air
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@82]    2 bnd Air///Water
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@86] msh_path /tmp/blyth/opticks/G4CXSimulateTest/RaindropRockAirWater2/CSGFoundry/meshname.txt msh.size 3
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@87]    0 msh Water_solid
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@87]    1 msh Air_solid
    2022-07-06 17:35:47.075 INFO  [2257351] [U4Recorder::init_CFBASE@87]    2 msh Rock_solid


    bflagdesc_(r[0,j])
     idx(     0) prd(b  0 p   0 i    0 o0 ii:    0)  TO               TO  :                                                   Rock_solid : 3ee28144 : Rock///Rock 
     idx(     0) prd(b  2 p   0 i    0 o0 ii:    0)  BT            TO|BT  :                                                   Rock_solid : 5499841d : Air///Water 
     idx(     0) prd(b  2 p   1 i    0 o0 ii:    0)  BT            TO|BT  :                                                    Air_solid : ec91a858 : Air///Water 
     idx(     0) prd(b  1 p   2 i    0 o0 ii:    0)  SA         TO|BT|SA  :                                                  Water_solid : 65ec719a : Rock//air_rock_bs/Air 



* discrepancy in the prim naming : seems to be in reversed order 

::

    In [1]: cf.primIdx_meshname_dict
    Out[1]: {0: 'Rock_solid', 1: 'Air_solid', 2: 'Water_solid'}


AHHA it is not a meshidx although it uses mesh names, it is a primIdx

::

    291     def make_primIdx_meshname_dict(self):
    292         """
    293         See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
    294         this method resolved an early naming bug 
    295 
    296         CSG/CSGPrim.h:: 
    297 
    298              95     PRIM_METHOD unsigned  meshIdx() const {           return q1.u.y ; }  // aka lvIdx
    299              96     PRIM_METHOD void   setMeshIdx(unsigned midx){     q1.u.y = midx ; }
    300 
    301         """
    302         d = {}
    303         for primIdx in range(len(self.prim)):
    304             midx = self.meshIdx (primIdx)      # meshIdx method with contiguous primIdx argument
    305             assert midx < len(self.meshname)
    306             mnam = self.meshname[midx]
    307             d[primIdx] = mnam
    308             #print("CSGFoundry:primIdx_meshname_dict primIdx %5d midx %5d meshname %s " % (primIdx, midx, mnam))
    309         pass
    310         return d


::

    epsilon:tests blyth$ ./CSGFoundryTest.sh 
    PLOG::EnvLevel adjusting loglevel by envvar   key CSGFoundry level INFO fallback DEBUG
    2022-07-06 18:03:17.416 INFO  [2282561] [*CSGFoundry::Load_@2358]  cfbase /tmp/blyth/opticks/G4CXSimulateTest/RaindropRockAirWater2 readable 1
    2022-07-06 18:03:17.417 INFO  [2282561] [CSGFoundry::load@2123] /tmp/blyth/opticks/G4CXSimulateTest/RaindropRockAirWater2/CSGFoundry
    2022-07-06 18:03:17.417 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     1 nj 3 nk 4 solid.npy
    2022-07-06 18:03:17.417 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     3 nj 4 nk 4 prim.npy
    2022-07-06 18:03:17.417 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     3 nj 4 nk 4 node.npy
    2022-07-06 18:03:17.418 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     3 nj 4 nk 4 tran.npy
    2022-07-06 18:03:17.418 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     3 nj 4 nk 4 itra.npy
    2022-07-06 18:03:17.418 INFO  [2282561] [CSGFoundry::loadArray@2448]  ni     1 nj 4 nk 4 inst.npy
    2022-07-06 18:03:17.421 INFO  [2282561] [*CSGFoundry::ELVString@2269]  elv_selection_ (null) elv (null)
    2022-07-06 18:03:17.421 INFO  [2282561] [CSGFoundry::getPrimName@214]  primIdx    0 midx 2 mname Rock_solid
    2022-07-06 18:03:17.421 INFO  [2282561] [CSGFoundry::getPrimName@214]  primIdx    1 midx 1 mname Air_solid
    2022-07-06 18:03:17.421 INFO  [2282561] [CSGFoundry::getPrimName@214]  primIdx    2 midx 0 mname Water_solid
    2022-07-06 18:03:17.421 INFO  [2282561] [test_getPrimName@221]  pname.size 3
    epsilon:tests blyth$ 





U4Recorder::getBoundary mimic Opticks boundary in G4
-------------------------------------------------------

::

    2022-07-06 14:55:21.909 INFO  [2029125] [U4Recorder::init@80] 0 : Rock///Rock
    2022-07-06 14:55:21.909 INFO  [2029125] [U4Recorder::init@80] 1 : Rock//air_rock_bs/Air
    2022-07-06 14:55:21.909 INFO  [2029125] [U4Recorder::init@80] 2 : Air///Water

::

    2022-07-06 14:56:16.672 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.674 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.676 INFO  [2030784] [U4Recorder::getBoundary@325]    1 : Rock//air_rock_bs/Air
    2022-07-06 14:56:16.678 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.680 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.682 INFO  [2030784] [U4Recorder::getBoundary@325]    1 : Rock//air_rock_bs/Air
    2022-07-06 14:56:16.684 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.687 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.689 INFO  [2030784] [U4Recorder::getBoundary@325]    1 : Rock//air_rock_bs/Air
    2022-07-06 14:56:16.691 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water
    2022-07-06 14:56:16.693 INFO  [2030784] [U4Recorder::getBoundary@325]    2 : Air///Water





DONE : get fast reproducible single (or small selection) photon running of B to work, little point with A currently as its so fast anyhow
---------------------------------------------------------------------------------------------------------------------------------------------

::

   PIDX=207 ./U4RecorderTest.sh run

* A:PIDX running means just output for that photon index
* B:PIDX running means just record stacks etc... for that photon index (making it much faster), and dump output too  

* writes to different fold when PIDX set
* currently writes original sized arrays with only one idx non-zero 

  * while wasteful to have so many zeros it is actually rather convenient, as can then address normally that index 
  * the primary reason for PIDX running is to dump Geant4 details that are not saved, like TransCoeff

::

    In [8]: a.base
    Out[8]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log/ALL'

    In [9]: b.base
    Out[9]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log/PIDX_207_'

    In [10]: a.photon[207]
    Out[10]: 
    array([[    3.475,   -22.598, -1000.   ,     7.552],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [   -0.988,    -0.152,     0.   ,   501.   ],
           [    0.   ,     0.   ,     0.   ,     0.   ]], dtype=float32)

    In [11]: b.photon[207]
    Out[11]: 
    array([[    3.475,   -22.598, -1000.   ,     7.552],
           [    0.   ,     0.   ,    -1.   ,     0.   ],
           [   -0.988,    -0.152,     0.   ,   501.   ],
           [    0.   ,     0.   ,     0.   ,     0.   ]], dtype=float32)

::

    In [1]: AB(207)
    Out[1]: 
    A : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log/ALL
    B : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log/PIDX_207_
    A(207) : TO BT BR BT SA                                                       B(207) : TO BT BR BT SA                                                       
           A.t : (10000, 48)                                                             B.t : (10000, 48)                                                      
          A.t2 : (10000, 48)                                                            B.t2 : (10000, 48)                                                      
           A.n : (10000,)                                                                B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                        B.ts : (10000, 48, 1)                                                   
          A.fs : (10000, 10, 29)                                                        B.fs : (10000, 48, 1)                                                   
         A.ts2 : (10000, 10, 29)                                                       B.ts2 : (10000, 48, 1)                                                   
     0 :     0.6107 :  3 : ScintDiscreteReset :                                    0 :     0.6107 :  3 : ScintDiscreteReset :                                   
     1 :     0.6644 :  4 : BoundaryDiscreteReset :                                 1 :     0.6644 :  4 : BoundaryDiscreteReset :                                
     2 :     0.6590 :  5 : RayleighDiscreteReset :                                 2 :     0.6590 :  5 : RayleighDiscreteReset :                                
     3 :     0.4623 :  6 : AbsorptionDiscreteReset :                               3 :     0.4623 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.3162 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :             4 :     0.3162 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.1116 :  8 : BoundaryDiDiTransCoeff :                                5 :     0.1116 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                
     6 :     0.4624 :  3 : ScintDiscreteReset :                                    6 :     0.4624 :  3 : ScintDiscreteReset :                                   
     7 :     0.5240 :  4 : BoundaryDiscreteReset :                                 7 :     0.5240 :  4 : BoundaryDiscreteReset :                                
     8 :     0.1806 :  5 : RayleighDiscreteReset :                                 8 :     0.1806 :  5 : RayleighDiscreteReset :                                
     9 :     0.4464 :  6 : AbsorptionDiscreteReset :                               9 :     0.4464 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5587 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            10 :     0.5587 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.9736 :  8 : BoundaryDiDiTransCoeff :                               11 :     0.9736 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                
    12 :     0.1517 :  3 : ScintDiscreteReset :                                   12 :     0.1517 :  3 : ScintDiscreteReset :                                   
    13 :     0.4271 :  4 : BoundaryDiscreteReset :                                13 :     0.4271 :  4 : BoundaryDiscreteReset :                                
    14 :     0.7832 :  5 : RayleighDiscreteReset :                                14 :     0.7832 :  5 : RayleighDiscreteReset :                                
    15 :     0.9705 :  6 : AbsorptionDiscreteReset :                              15 :     0.9705 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                
    16 :     0.2868 :  3 : ScintDiscreteReset :                                   16 :     0.2868 :  3 : ScintDiscreteReset :                                   
    17 :     0.8723 :  4 : BoundaryDiscreteReset :                                17 :     0.8723 :  4 : BoundaryDiscreteReset :                                
    18 :     0.1749 :  5 : RayleighDiscreteReset :                                18 :     0.1749 :  5 : RayleighDiscreteReset :                                
    19 :     0.0048 :  6 : AbsorptionDiscreteReset :                              19 :     0.0048 :  6 : AbsorptionDiscreteReset :                              
    20 :     0.8760 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            20 :     0.8760 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    21 :     0.9752 :  8 : BoundaryDiDiTransCoeff :                               21 :     0.9752 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                
    22 :     0.6843 :  3 : ScintDiscreteReset :                                   22 :     0.6843 :  3 : ScintDiscreteReset :                                   
    23 :     0.9146 :  4 : BoundaryDiscreteReset :                                23 :     0.9146 :  4 : BoundaryDiscreteReset :                                
    24 :     0.6236 :  5 : RayleighDiscreteReset :                                24 :     0.6236 :  5 : RayleighDiscreteReset :                                
    25 :     0.7684 :  6 : AbsorptionDiscreteReset :                              25 :     0.7684 :  6 : AbsorptionDiscreteReset :                              
    26 :     0.2045 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            26 :     0.2045 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    27 :     0.6549 :  9 : AbsorptionEffDetect :                                  27 :     0.6549 :  9 : AbsorptionEffDetect :                                  
    28 :     0.0000 :  0 : Unclassified :                                         28 :     0.0000 :  0 : Unclassified :                                         
    29 :     0.0000 :  0 : Unclassified :                                         29 :     0.0000 :  0 : Unclassified :                                         






TODO : reduce truncation
---------------------------

TODO: as not aligning reemission can switch from 5 bits to 4 and hence up from 48 slots to 64 slots without increasing storage

AHHA some of issue could be from truncation, 48 is not enough slots for the longer histories of more complicated geom:: 

    In [4]: A.t[0]
    Out[4]: array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2], dtype=uint8)

    In [5]: A.t.shape
    Out[5]: (10000, 48)

::

    In [11]: wt = np.where( A.t[:,47] != 0 )[0] ; len(wt)
    Out[11]: 368

    In [12]: seqhis_(a.seq[wt,0])   ## 9 or 10 point seqhis are getting truncated
    Out[12]: 
    ['TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BT BR BT BT BT BT',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',


TODO : add boundary + identity to B:photon/record flags 
---------------------------------------------------------------------

::

    In [7]: a.record.view(np.int32)[0,:,3]
    Out[7]: 
    array([[4096,    0,    0, 4096],
           [2048,    0,    0, 6144],
           [2048,    0,    0, 6144],
           [2048,    0,    0, 6144],
           [2048,    0,    0, 6144],
           [2048,    0,    0, 6144],
           [ 128,    0,    0, 6272],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0]], dtype=int32)

    In [9]: a.photon.view(np.int32)[0,3]
    Out[9]: array([ 128,    0,    0, 6272], dtype=int32)



TODO : ADD B:side boundary/identity 
-------------------------------------------

boundaries
   boundaries have names based on material and surface names so the B side
   can access this set of names from the A side at initialization and hence derive a boundary index 
   from a live set of Geant4 pre/post points that straddle the boundary    

   * can detect CFBASE envvar to know to pick where to load the bnd_names from 
   * NP::ReadNames("$CFBASE/CSGFoundry/SSim/bnd_names.txt" 

identity 
   hmm: what exactly is the A side identity : primIdx probably so that is solid/lv index ? 
   simtrace plotting uses this for the keys, see cx/tests/CSGOptiXSimtraceTest.py

   * G4 accessing the volume : its like what happens with a hit. Possible but not very nice. 
   * but with simple geometries the boundary probably sufficient for debugging

* DONE : start by interpreting/dumping the A boundaries+identity then work out how to reproduce them Geant4 side 
* DONE : for this will need to save the GGeo/CSGFoundry geocache and grab it in orde to hookup the actual geometry to the python machinery 


::

    In [32]: boundary___(r[0])
    Out[32]: array([0, 2, 3, 3, 3, 3, 3, 2, 1, 0], dtype=uint32)

    In [36]: seqhis_(t.seq[0,0])
    Out[36]: 'TO BT BT BT BR BT BT BT SA'


Capture this into XFold::

    In [1]: A[0]                                                                                                                    
    Out[1]: 
    A : /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL
    A(0) : TO BT BT BT BR BT BT BT SA
    - Water///Pyrex                            hama_body_solid_1_4                               
    - Pyrex///Vacuum                           hama_inner2_solid_1_4                             
    - Pyrex///Vacuum                           hama_inner1_solid_I                               
    + Pyrex///Vacuum                           hama_inner1_solid_I                               
    + Pyrex///Vacuum                           hama_inner1_solid_I                               
    + Pyrex///Vacuum                           hama_inner2_solid_1_4                             
    + Water///Pyrex                            hama_body_solid_1_4                               
    + Rock//water_rock_bs/Water                Water_solid                                       

    In [2]:                         


G4CXSimulateTest.cc::

     41     else if(SSys::hasenvvar("GEOM"))
     42     {
     43         gx.setGeometry( U4VolumeMaker::PV() );
     44         assert(gx.fd);
     45 
     46         const char* cfdir = SPath::Resolve("$DefaultOutputDir/CSGFoundry", DIRPATH);
     47         gx.fd.write(cfdir);
     48     }

::

    gx
    ./gxs.sh grab 
    ...

    == ../bin/rsync.sh tto /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log jpg mp4 npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/solid.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/prim.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/node.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/tran.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/itra.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/inst.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/SSim/bnd.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/SSim/propcom.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/SSim/optical.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/photon.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/genstep.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/record.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/rec.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/seq.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/prd.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/tag.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/seed.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/inphoton.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/domain.npy
    /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/ALL/flat.npy

    epsilon:SSim blyth$ cat /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/SSim/bnd_names.txt
    Rock///Rock
    Rock//water_rock_bs/Water
    Water///Pyrex
    Pyrex///Vacuum

    epsilon:SSim blyth$ cat /tmp/blyth/opticks/G4CXSimulateTest/hama_body_log/CSGFoundry/meshname.txt 
    hama_inner1_solid_I
    hama_inner2_solid_1_4
    hama_body_solid_1_4
    Water_solid
    Rock_solid
    epsilon:SSim blyth$ 


The sctx::point persists the sphoton but where is p.flag/p.boundary set::

     84 SCTX_METHOD void sctx::point(int bounce)
     85 {
     86     if(evt->record && bounce < evt->max_record) evt->record[evt->max_record*idx+bounce] = p ;
     87     if(evt->rec    && bounce < evt->max_rec)    evt->add_rec( rec, idx, bounce, p );    // this copies into evt->rec array 
     88     if(evt->seq    && bounce < evt->max_seq)    seq.add_nibble( bounce, p.flag(), p.boundary() );
     89 }
     90 SCTX_METHOD void sctx::trace(int bounce)
     91 {
     92     if(evt->prd) evt->prd[evt->max_prd*idx+bounce] = *prd ;
     93 }

::

    202 void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
    203 {
    204     const G4StepPoint* pre = step->GetPreStepPoint() ;
    205     const G4StepPoint* post = step->GetPostStepPoint() ;
    206     const G4Track* track = step->GetTrack();
    207 
    208     spho label = U4Track::Label(track);
    209     assert( label.isDefined() );
    210     if(!Enabled(label)) return ;  // early debug  
    211 
    212     //LOG(info) << " label.id " << label.id << " " << U4Process::Desc() ; 
    213 
    214     SEvt* sev = SEvt::Get();
    215     sev->checkPhotonLineage(label);
    216     sphoton& current_photon = sev->current_ctx.p ;
    217 
    218     bool first_point = current_photon.flagmask_count() == 1 ;  // first_point when single bit in the flag from genflag set in beginPhoton
    219     if(first_point)
    220     {
    221         U4StepPoint::Update(current_photon, pre);
    222         sev->pointPhoton(label);  // saves SEvt::current_photon/rec/record/prd into sevent 
    223     }
    224 
    225     unsigned flag = U4StepPoint::Flag(post) ;
    226     if( flag == 0 ) LOG(error) << " ERR flag zero : post " << U4StepPoint::Desc(post) ;
    227     assert( flag > 0 );
    228 


    229     unsigned boundary = 0 ;   // TODO: rustle up these 
    230     unsigned identity = 0 ;
    231     
    232     if( flag == NAN_ABORT )
    233     {   
    234         LOG(LEVEL) << " skip post saving for StepTooSmall label.id " << label.id  ;
    235     }
    236     else
    237     {   
    238         G4TrackStatus tstat = track->GetTrackStatus();
    239         Check_TrackStatus_Flag(tstat, flag);
    240         
    241         U4StepPoint::Update(current_photon, post);
    242         current_photon.set_flag( flag );
    243         current_photon.set_boundary( boundary);
    244         current_photon.identity = identity ;
    245         
    246         sev->pointPhoton(label);         // save SEvt::current_photon/rec/seq/prd into sevent 
    247     }
    248     U4Process::ClearNumberOfInteractionLengthLeft(*track, *step);
    249 }



::

     80     unsigned boundary_flag ;
     81     unsigned identity ;
     82     unsigned orient_idx ;
     83     unsigned flagmask ;


     97     SPHOTON_METHOD void     set_flag(unsigned flag) {         boundary_flag = ( boundary_flag & 0xffff0000u ) | ( flag & 0xffffu ) ; flagmask |= flag ;  } // clear flag bits then set them  
     98     SPHOTON_METHOD void     set_boundary(unsigned boundary) { boundary_flag = ( boundary_flag & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ; }        // clear boundary bits then set them 


"B"::

    In [15]: a.base
    Out[15]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log/ALL'

    In [14]: np.all( a.record[:,:,3,1].view(np.uint32)  == 0 )
    Out[14]: True


    In [17]: a.record.view(np.int32)[207,:,3]
    Out[17]: 
    array([[4096,    0,  207, 4096],
           [2048,    0,  207, 6144],
           [1024,    0,  207, 7168],
           [2048,    0,  207, 7168],
           [ 128,    0,  207, 7296],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0],
           [   0,    0,    0,    0]], dtype=int32)


* looks like only flag/idx/flagmask being set : so no identity or boundary for B 


enum align checking by converting a sequence of tags to a string for each idx to compare 
--------------------------------------------------------------------------------------------

::

    In [17]: A.t[2]
    Out[17]: array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)

    In [18]: B.t2[2]
    Out[18]: array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)

    In [20]: A.ts[2]
    Out[20]: 
    array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    In [21]: B.ts2[2]
    Out[21]: 
    array([[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)


Numpy way to do::

    In [25]: for i in range(len(A.t)): 
        ...:     if np.all( A.t[i] == B.t2[i]): print(i)  
        ...:                                                                                                                                                                                                  
    5
    36
    39
    54
    64
    75

Use the fact that the enum are small numbers so view the full history as string and compare those::

    A.t[9853].view("|S48") == B.t2[9853].view("|S48")  

    In [34]: we = np.where( A.t.view("|S48") == B.t2.view("|S48") )[0] ; len(we)
    Out[34]: 644

    In [37]: np.all( a.seq[we,0] == b.seq[we,0] )   ## history aligned for those as they should be 
    Out[37]: True


The 644/10k that are enum aligned did not go thru the middle::

    In [40]: o = cuss( a.seq[we,0], we )

    In [41]: o
    Out[41]: 
    CUSS([['w0', '                TO BT BR BT SA', '          576461', '             348'],
          ['w1', '                         TO AB', '              77', '             211'],
          ['w2', '                      TO BT AB', '            1229', '              31'],
          ['w3', '                      TO BR SA', '            2237', '              20'],
          ['w4', '                      TO SC SA', '            2157', '              17'],
          ['w5', '                TO BT BR BT AB', '          314317', '              12'],
          ['w6', '          TO SC BT BT BT BT SA', '       147639405', '               1'],
          ['w7', '          TO SC BT BT BT BT AB', '        80530541', '               1'],
          ['w8', '             TO BT BR BT SC SA', '         8833997', '               1'],
          ['w9', '                   TO BT BR AB', '           19405', '               1'],
          ['w10', '                      TO SC AB', '            1133', '               1']], dtype=object)


Check the one of the aligned with a BR::

    In [19]: AB(we[17])
    Out[19]: 
    A(207) : TO BT BR BT SA                                                                 B(207) : TO BT BR BT SA                                                       
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
          A.t2 : (10000, 48)                                                                      B.t2 : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 9, 29)                                                                   B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 9, 29)                                                                   B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 9, 29)                                                                  B.ts2 : (10000, 10, 29)                                                  
     0 :     0.6107 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.6107 :  3 : ScintDiscreteReset :                                   
     1 :     0.6644 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.6644 :  4 : BoundaryDiscreteReset :                                
     2 :     0.6590 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.6590 :  5 : RayleighDiscreteReset :                                
     3 :     0.4623 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.4623 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.3162 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.3162 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.1116 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.1116 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.4624 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.4624 :  3 : ScintDiscreteReset :                                   
     7 :     0.5240 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.5240 :  4 : BoundaryDiscreteReset :                                
     8 :     0.1806 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.1806 :  5 : RayleighDiscreteReset :                                
     9 :     0.4464 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.4464 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5587 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                10 :     0.5587 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.9736 :  6 :     at_ref : u_reflect > TransCoeff                              11 :     0.9736 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    12 :     0.1517 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           12 :     0.1517 :  3 : ScintDiscreteReset :                                   
    13 :     0.4271 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           13 :     0.4271 :  4 : BoundaryDiscreteReset :                                
    14 :     0.7832 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            14 :     0.7832 :  5 : RayleighDiscreteReset :                                
    15 :     0.9705 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            15 :     0.9705 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                          
    16 :     0.2868 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           16 :     0.2868 :  3 : ScintDiscreteReset :                                   
    17 :     0.8723 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           17 :     0.8723 :  4 : BoundaryDiscreteReset :                                
    18 :     0.1749 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            18 :     0.1749 :  5 : RayleighDiscreteReset :                                
    19 :     0.0048 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            19 :     0.0048 :  6 : AbsorptionDiscreteReset :                              
    20 :     0.8760 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                20 :     0.8760 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    21 :     0.9752 :  6 :     at_ref : u_reflect > TransCoeff                              21 :     0.9752 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    22 :     0.6843 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           22 :     0.6843 :  3 : ScintDiscreteReset :                                   
    23 :     0.9146 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           23 :     0.9146 :  4 : BoundaryDiscreteReset :                                
    24 :     0.6236 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            24 :     0.6236 :  5 : RayleighDiscreteReset :                                
    25 :     0.7684 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            25 :     0.7684 :  6 : AbsorptionDiscreteReset :                              
    26 :     0.2045 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                26 :     0.2045 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    27 :     0.6549 :  7 :    sf_burn : qsim::propagate_at_surface burn                     27 :     0.6549 :  9 : AbsorptionEffDetect :                                  
    28 :     0.0000 :  0 :      undef : undef                                               28 :     0.0000 :  0 : Unclassified :                                         
    29 :     0.0000 :  0 :      undef : undef                                               29 :     0.0000 :  0 : Unclassified :                                         






Check back with simple geom, shows have full enum alignment with that::

    a.base                                             : /tmp/blyth/opticks/G4CXSimulateTest/RaindropRockAirWater2
    b.base                                             : /tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/RaindropRockAirWater2

    In [1]: we = np.where( A.t.view("|S48") == B.t2.view("|S48") )[0] ; len(we)
    Out[1]: 10000





General Look
-----------------

Maybe need microstep skipping (or skipping virtual skins) like did previously.

Histories of first 10::

    In [9]: seqhis_(a.seq[:10,0])
    Out[9]: 
    ['TO BT BT BT BR BT BT BT SA',
     'TO BT BT AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BR BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA']

    In [10]: seqhis_(b.seq[:10,0])
    Out[10]: 
    ['TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA']

2/TO BT BT [BT] BT BT SA/history matched but time off from mid-point/probably degenerate surfaces mean using wrong groupvel::

    In [21]: a.record[2,:7] - b.record[2,:7]
    Out[21]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[-0.   ,  0.   ,  0.   ,  0.301],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[-0.004,  0.002,  0.   ,  0.302],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)


point-to-point position time deltas within A and B::

    In [24]: a.record[2,1:7,0] - a.record[2,0:6,0]
    Out[24]: 
    array([[  0.   ,   0.   , 806.775,   3.728],
           [  0.   ,   0.   ,   5.   ,   0.025],
           [  0.   ,   0.   , 178.225,   *0.896*],
           [  0.   ,   0.   , 184.558,   0.616],
           [  0.071,  -0.044,   5.002,   0.025],
           [  9.177,  -5.715, 810.44 ,   3.746]], dtype=float32)

    In [25]: b.record[2,1:7,0] - b.record[2,0:6,0]
    Out[25]: 
    array([[  0.   ,   0.   , 806.775,   3.728],
           [  0.   ,   0.   ,   5.   ,   0.025],
           [  0.   ,   0.   , 178.225,   *0.594*],
           [  0.   ,   0.   , 184.558,   0.616],
           [  0.071,  -0.044,   5.002,   0.025],
           [  9.181,  -5.717, 810.44 ,   3.745]], dtype=float32)


4/TO BT BT [BT] BT BT SA/history matched but time off from mid-point::

    In [20]: a.record[4,:7] - b.record[4,:7]
    Out[20]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],  ## time off from middle point TO BT BT [BT] BT BT SA
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.   ,  0.301],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [-0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   ,  0.   ,  0.301],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.013,  0.014,  0.   ,  0.303],
            [ 0.   ,  0.   , -0.   ,  0.   ],
            [ 0.   , -0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)


5/TO AB::

    In [18]: a.record[5,:2] - b.record[5,:2]
    Out[18]: 
    array([[[ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]],

           [[ 0.   ,  0.   , -0.003, -0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   ,  0.   ,  0.   ],
            [ 0.   ,  0.   , -0.   ,  0.   ]]], dtype=float32)



Checking those with matched histories shows no BR on internal layers in first 100 anyhow::

    In [14]: seqhis_( b.seq[wm[:100],0] )
    Out[14]: 
    ['TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO AB',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',
     'TO BT BT BT BT BT SA',




Scripted interleaving with sysrap/ABR.py
-------------------------------------------

DONE: script such interleaving "AB(0)" and move the result : BT/BR/... alongside the decision random

* sysrap/ABR.py presents repr of two objects side-by-side 

Developed with the fully aligned raindrop geom::

    In [2]: AB(4)
    Out[2]: 
    A(4) : TO BT BT SA                                                                      B(4) : TO BT BT SA                                                            
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.9251 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.9251 :  3 : ScintDiscreteReset :                                   
     1 :     0.0530 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.0530 :  4 : BoundaryDiscreteReset :                                
     2 :     0.1631 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.1631 :  5 : RayleighDiscreteReset :                                
     3 :     0.8897 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.8897 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.5666 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.5666 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.2414 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.2414 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.4937 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.4937 :  3 : ScintDiscreteReset :                                   
     7 :     0.3212 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.3212 :  4 : BoundaryDiscreteReset :                                
     8 :     0.0786 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.0786 :  5 : RayleighDiscreteReset :                                
     9 :     0.1479 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.1479 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5987 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                10 :     0.5987 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.4265 :  6 :     at_ref : u_reflect > TransCoeff                              11 :     0.4265 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    12 :     0.2435 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           12 :     0.2435 :  3 : ScintDiscreteReset :                                   
    13 :     0.4892 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           13 :     0.4892 :  4 : BoundaryDiscreteReset :                                
    14 :     0.4095 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            14 :     0.4095 :  5 : RayleighDiscreteReset :                                
    15 :     0.6676 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            15 :     0.6676 :  6 : AbsorptionDiscreteReset :                              
    16 :     0.6269 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                16 :     0.6269 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    17 :     0.2769 :  7 :    sf_burn : qsim::propagate_at_surface burn                     17 :     0.2769 :  9 : AbsorptionEffDetect :                                  
    18 :     0.0000 :  0 :      undef : undef                                               18 :     0.0000 :  0 : Unclassified :                                         
    19 :     0.0000 :  0 :      undef : undef                                               19 :     0.0000 :  0 : Unclassified :                                         


Normally there is one less consumption clump than there are step points. But when there is a BR 
there is an extra consumption clump from the Geant4 StepTooSmall and Opticks mimicking that with burns to retain alignment::

    In [5]: AB(3)
    Out[5]: 
    A(3) : TO BR SA                                                                         B(3) : TO BR SA                                                               
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.9690 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.9690 :  3 : ScintDiscreteReset :                                   
     1 :     0.4947 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.4947 :  4 : BoundaryDiscreteReset :                                
     2 :     0.6734 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.6734 :  5 : RayleighDiscreteReset :                                
     3 :     0.5628 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.5628 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.1202 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.1202 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.9765 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.9765 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.1358 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.1358 :  3 : ScintDiscreteReset :                                   
     7 :     0.5890 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.5890 :  4 : BoundaryDiscreteReset :                                
     8 :     0.4906 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.4906 :  5 : RayleighDiscreteReset :                                
     9 :     0.3284 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.3284 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                          
    10 :     0.9114 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           10 :     0.9114 :  3 : ScintDiscreteReset :                                   
    11 :     0.1907 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           11 :     0.1907 :  4 : BoundaryDiscreteReset :                                
    12 :     0.9637 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            12 :     0.9637 :  5 : RayleighDiscreteReset :                                
    13 :     0.8976 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            13 :     0.8976 :  6 : AbsorptionDiscreteReset :                              
    14 :     0.6243 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                14 :     0.6243 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    15 :     0.7102 :  7 :    sf_burn : qsim::propagate_at_surface burn                     15 :     0.7102 :  9 : AbsorptionEffDetect :                                  
    16 :     0.0000 :  0 :      undef : undef                                               16 :     0.0000 :  0 : Unclassified :                                         
    17 :     0.0000 :  0 :      undef : undef                                               17 :     0.0000 :  0 : Unclassified :          


    In [8]: AB(36)
    Out[8]: 
    A(36) : TO BT BR BT SA                                                                  B(36) : TO BT BR BT SA                                                        
           A.t : (10000, 48)                                                                       B.t : (10000, 48)                                                      
           A.n : (10000,)                                                                          B.n : (10000,)                                                         
          A.ts : (10000, 10, 29)                                                                  B.ts : (10000, 10, 29)                                                  
          A.fs : (10000, 10, 29)                                                                  B.fs : (10000, 10, 29)                                                  
         A.ts2 : (10000, 10, 29)                                                                 B.ts2 : (10000, 10, 29)                                                  
     0 :     0.2405 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            0 :     0.2405 :  3 : ScintDiscreteReset :                                   
     1 :     0.4503 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            1 :     0.4503 :  4 : BoundaryDiscreteReset :                                
     2 :     0.2029 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             2 :     0.2029 :  5 : RayleighDiscreteReset :                                
     3 :     0.5092 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             3 :     0.5092 :  6 : AbsorptionDiscreteReset :                              
     4 :     0.2154 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                 4 :     0.2154 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
     5 :     0.1141 :  6 :     at_ref : u_reflect > TransCoeff                               5 :     0.1141 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
     6 :     0.3870 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn            6 :     0.3870 :  3 : ScintDiscreteReset :                                   
     7 :     0.8183 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn            7 :     0.8183 :  4 : BoundaryDiscreteReset :                                
     8 :     0.2030 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering             8 :     0.2030 :  5 : RayleighDiscreteReset :                                
     9 :     0.7006 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption             9 :     0.7006 :  6 : AbsorptionDiscreteReset :                              
    10 :     0.5327 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                10 :     0.5327 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    11 :     0.9862 :  6 :     at_ref : u_reflect > TransCoeff                              11 :     0.9862 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    12 :     0.5105 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           12 :     0.5105 :  3 : ScintDiscreteReset :                                   
    13 :     0.3583 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           13 :     0.3583 :  4 : BoundaryDiscreteReset :                                
    14 :     0.9380 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            14 :     0.9380 :  5 : RayleighDiscreteReset :                                
    15 :     0.4586 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            15 :     0.4586 :  6 : AbsorptionDiscreteReset :                              
                                                                                                                                                                          
    16 :     0.9189 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           16 :     0.9189 :  3 : ScintDiscreteReset :                                   
    17 :     0.1870 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           17 :     0.1870 :  4 : BoundaryDiscreteReset :                                
    18 :     0.2109 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            18 :     0.2109 :  5 : RayleighDiscreteReset :                                
    19 :     0.9003 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            19 :     0.9003 :  6 : AbsorptionDiscreteReset :                              
    20 :     0.0704 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                20 :     0.0704 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    21 :     0.7765 :  6 :     at_ref : u_reflect > TransCoeff                              21 :     0.7765 :  8 : BoundaryDiDiTransCoeff :                               
                                                                                                                                                                          
    22 :     0.3422 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn           22 :     0.3422 :  3 : ScintDiscreteReset :                                   
    23 :     0.1178 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn           23 :     0.1178 :  4 : BoundaryDiscreteReset :                                
    24 :     0.5520 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering            24 :     0.5520 :  5 : RayleighDiscreteReset :                                
    25 :     0.3090 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption            25 :     0.3090 :  6 : AbsorptionDiscreteReset :                              
    26 :     0.0165 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd                26 :     0.0165 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :            
    27 :     0.4159 :  7 :    sf_burn : qsim::propagate_at_surface burn                     27 :     0.4159 :  9 : AbsorptionEffDetect :                                  
    28 :     0.0000 :  0 :      undef : undef                                               28 :     0.0000 :  0 : Unclassified :                                         
    29 :     0.0000 :  0 :      undef : undef                                               29 :     0.0000 :  0 : Unclassified :                                         




Manually interleaving A(0) B(0) shows where alignment is lost
---------------------------------------------------------------




::

    In [29]: A(0)
    Out[29]: 
    A(0) : TO BT BT BT BR BT BT BT SA
           A.t : (10000, 48) 
           A.n : (10000,) 
          A.ts : (10000, 9, 29) 
          A.fs : (10000, 9, 29) 
         A.ts2 : (10000, 9, 29) 

    B(0) : TO BT BT BT BT BT SA
           B.t : (10000, 48) 
           B.n : (10000,) 
          B.ts : (10000, 10, 29) 
          B.fs : (10000, 10, 29) 
         B.ts2 : (10000, 10, 29) 


     0 :     0.7402 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     1 :     0.4385 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     2 :     0.5170 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     3 :     0.1570 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
     4 :     0.0714 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
     5 :     0.4625 :  6 :     at_ref : u_reflect > TransCoeff 

     0 :     0.7402 :  3 : ScintDiscreteReset :  
     1 :     0.4385 :  4 : BoundaryDiscreteReset :  
     2 :     0.5170 :  5 : RayleighDiscreteReset :  
     3 :     0.1570 :  6 : AbsorptionDiscreteReset :  
     4 :     0.0714 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
     5 :     0.4625 :  8 : BoundaryDiDiTransCoeff :  



     6 :     0.2276 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
     7 :     0.3294 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
     8 :     0.1441 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
     9 :     0.1878 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    10 :     0.9154 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    11 :     0.5401 :  6 :     at_ref : u_reflect > TransCoeff 

     6 :     0.2276 :  3 : ScintDiscreteReset :  
     7 :     0.3294 :  4 : BoundaryDiscreteReset :  
     8 :     0.1441 :  5 : RayleighDiscreteReset :  
     9 :     0.1878 :  6 : AbsorptionDiscreteReset :  
    10 :     0.9154 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    11 :     0.5401 :  8 : BoundaryDiDiTransCoeff :  



    12 :     0.9747 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    13 :     0.5475 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    14 :     0.6532 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    15 :     0.2302 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    16 :     0.3389 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    17 :     0.7614 :  6 :     at_ref : u_reflect > TransCoeff 

    12 :     0.9747 :  3 : ScintDiscreteReset :  
    13 :     0.5475 :  4 : BoundaryDiscreteReset :  
    14 :     0.6532 :  5 : RayleighDiscreteReset :  
    15 :     0.2302 :  6 : AbsorptionDiscreteReset :  

    ##  ALIGNMENT LOST HERE : THATS MAYBE A StepTooSmall ?


    18 :     0.5457 :  1 :     to_sci : qsim::propagate_to_boundary u_to_sci burn 
    19 :     0.9703 :  2 :     to_bnd : qsim::propagate_to_boundary u_to_bnd burn 
    20 :     0.2112 :  3 :     to_sca : qsim::propagate_to_boundary u_scattering 
    21 :     0.9469 :  4 :     to_abs : qsim::propagate_to_boundary u_absorption 
    22 :     0.5530 :  5 : at_burn_sf_sd : at_boundary_burn at_surface ab/sd  
    23 :     0.9776 :  6 :     at_ref : u_reflect > TransCoeff 


    16 :     0.3389 :  3 : ScintDiscreteReset :  
    17 :     0.7614 :  4 : BoundaryDiscreteReset :  
    18 :     0.5457 :  5 : RayleighDiscreteReset :  
    19 :     0.9703 :  6 : AbsorptionDiscreteReset :  
    20 :     0.2112 :  7 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
    21 :     0.9469 :  8 : BoundaryDiDiTransCoeff :  





TODO: get gxr working to visualize this
-------------------------------------------

 
