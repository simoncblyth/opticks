CSGClone_CSGSelection_Prim_selection
=======================================

center_extent z-position difference in r4, r6, r8

* fixed by skipping bb inclusion for CSG_ZERO nodes


Debug Prim Selection
-----------------------

Test on laptop with simple g4ok/G4OKVolumeTest.sh geometry::

   g4ok
    ./G4OKVolumeTest.sh 

   cg
   ./run.sh 

   c
   ./CSGPrimTest.sh asis 




::

    ELV=t EYE=2,2,2 TMIN=3 ./cxr_overview.sh    
    ## ELV=t not-nothing : which means all works as expected 
    ## showing cut open world box with small JustOrb and BoxMinusOrb inside


    2022-03-15 15:51:16.232 INFO  [5575168] [CSGOptiXRenderTest::initFD@190]  ELV         t   3 : 111
    CSGFoundry::descELV elv.num_bits 3 include 3 exclude 0
    INCLUDE:3

      0:JustOrb
      1:BoxMinusOrb
      2:World_solid
    EXCLUDE:0

    2022-03-15 15:59:18.974 INFO  [5583912] [*CSGCopy::Select@35] 
    src:CSGFoundry  num_total 1 num_solid 1 num_prim 28 num_node 54 num_plan 0 num_tran 41 num_itra 41 num_inst 1 ins 0 gas 0 ias 0 meshname 3 mmlabel 1
    dst:CSGFoundry  num_total 1 num_solid 1 num_prim 28 num_node 54 num_plan 0 num_tran 41 num_itra 41 num_inst 1 ins 0 gas 0 ias 0 meshname 3 mmlabel 1



::


    EYE=2,2,2 ELV=0 TMIN=0 ./cxr_overview.sh 
        blank render  

    EYE=2,2,2 ELV=1 TMIN=0 ./cxr_overview.sh 
        blank render  

    EYE=2,2,2 ELV=2 TMIN=3 ./cxr_overview.sh 
        get expected empty world box

    EYE=2,2,2 ELV=t2 TMIN=3 ./cxr_overview.sh 
        scatter of small boxes without the world box, no orbs, no boxminusorb

    ELV=t CSGPrimSpecTest  
    ELV=0,1 CSGPrimSpecTest  
    ELV=t0 CSGPrimSpecTest     
         no surprises, get expected prim bbox 

    EYE=2,2,2 ELV=2,0 TMIN=3 ./cxr_overview.sh 
        interesting : this gives cut open world box with a single orb at origin (not expected grid of orb) 
        suggests missing transforms ?

    EYE=2,2,2 ELV=t1 TMIN=3 ./cxr_overview.sh 
         get same as ELV=2,0  


    EYE=2,2,2 ELV=2,1 TMIN=3 ./cxr_overview.sh 
        hmm : now get single small box (not boxminusorb as it should be) offset from origin within the world box

    EYE=2,2,2 ELV=t0 TMIN=3 ./cxr_overview.sh 
         gives same as ELV=2,1




    EYE=2,2,2 ELV=2,1,0 TMIN=3 ./cxr_overview.sh 
        here back to normal 

    
    








::

    epsilon:CSG blyth$ ./CSGPrimTest.sh asis
                     arg : asis 
    OPTICKS_GEOCACHE_HOOKUP_ARG : asis 
             OPTICKS_KEY : G4OKVolumeTest.X4PhysicalVolume.World_pv.454372a9f3c659bed5168603f4a26a22 
    INFO:__main__:arg   : asis 
    INFO:__main__:kd    : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1 
    INFO:__main__:cfdir : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry 
    INFO:opticks.CSG.CSGFoundry:load /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry 
    /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry
    min_stamp:2022-03-09 09:57:13.463482
    max_stamp:2022-03-09 09:57:13.466192
    age_stamp:6 days, 6:03:43.480930
             node :           (54, 4, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/node.npy 
             itra :           (41, 4, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/itra.npy 
         meshname :                 (3,)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/meshname.txt 
             meta :                 (1,)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/meta.txt 
             icdf :         (0, 4096, 1)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/icdf.npy 
          mmlabel :                 (1,)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/mmlabel.txt 
             tran :           (41, 4, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/tran.npy 
             inst :            (1, 4, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/inst.npy 
              bnd :    (1, 4, 2, 761, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/bnd.npy 
            solid :            (1, 3, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/solid.npy 
             prim :           (28, 4, 4)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/prim.npy 
         bnd_meta :                 (1,)  : /usr/local/opticks/geocache/G4OKVolumeTest_World_pv_g4live/g4ok_gltf/454372a9f3c659bed5168603f4a26a22/1/CSG_GGeo/CSGFoundry/bnd_meta.txt 


     all_ridxs: [0]   ridxs:[0]   nmame_skip:Flange   geocache_hookup_arg:asis 

     ridx :  0   ridx_prims.shape (28, 4, 4) 
     u_mx c_mx  nnd :                        unique midx prim counts and meshname  : prs.shape 
        0   14    1 :                                                      JustOrb :           (14, 4, 4) : [-600. -600. -600. -400. -400. -400.    0.    0.]  
        1   13    3 :                                                  BoxMinusOrb :           (13, 4, 4) : [-600. -600. -100. -400. -400.  100.    0.    0.]  
        2    1    1 :                                                  World_solid :            (1, 4, 4) : [-1500. -1500. -1500.  1500.  1500.  1500.     0.     0.]  
     skip:0  nmame_skip:Flange 

    In [1]:                              





    ELV=t2 EYE=2,2,2 TMIN=3 ./cxr_overview.sh 
    ## expecting to exlcude the world box but instead get multiple boxes  
    ## suggests problem with instancing    

    INCLUDE:2

      0:JustOrb
      1:BoxMinusOrb
    EXCLUDE:1

      2:World_solid





