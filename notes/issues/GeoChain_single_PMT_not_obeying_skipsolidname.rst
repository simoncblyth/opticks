GeoChain_single_PMT_not_obeying_skipsolidname
================================================

Issue : cxs render shows outer PMT solid only that appears to not have the horizontals
----------------------------------------------------------------------------------------

* assume cause is a failure to skip the degenerate body solid
  due to the highly abnormal GeoChain single PMT only geometry
  
* there could be problems from that, but subsequently found an ellipsoid stomping on scale transform bug, now
  fixed as described in :doc:`ellipsoid_not_maintaining_shape_within_boolean_combination`

* following the fix the intersects with body_phys follow the expected shape, BUT only 
  one solid showing : want to see the multiple offset layers 

* the one solid was pilot error, need to ISEL select, showing multiple layers, but with impinging sombrero::


::

    # build PMTSim lib, providing standalone PV creation 
    jps    # cd ~/j/PMTSim
    om

    # build GeoChain which links with PMTSIm lib 
    gc     # cd ~/opticks/GeoChain
    om

    # run GeoChain with GeoChainVolumeTest creating the PV and running it through the conversions
    # script now auto selects the appropriate binary based on the geometry name
    GEOM=body_phys ./run.sh   

    # cxs 2d intersect render
    cx 
    ./b7
    GEOM=body_phys ./cxs.sh 

    # on laptop grap intersects and display locally 
    cx 
    ./grab.sh 
    GEOM=body_phys ./cxs.sh 

    GEOM=body_phys ISEL=0,1,2,3,4,5 ./cxs.sh  


Mysterious innards : maybe from lack of --gparts_transform_offset in GeoChainVolumeTest.cc
----------------------------------------------------------------------------------------------

* /tmp/blyth/opticks/CSGOptiX/CSGOptiXSimulateTest/body_phys/figs/positions_plt_0,1,2,3,4,5.png

Force add the option::

     39     const char* argforced = "--allownokey --gparts_transform_offset" ;
     40     Opticks ok(argc, argv, argforced);
     41     ok.configure();


::

    CXS=body_phys ISEL=0 ./cxs.sh     # red  : expected PMT outer shape
    CXS=body_phys ISEL=1 ./cxs.sh     # green: mysterious innards
    CXS=body_phys ISEL=2 ./cxs.sh     # blue : expected iseects on upper hemi-ellipsoid


::

    In [1]: ph.ubnd                                                                                                                                                                                     
    Out[1]: array([0, 1, 2], dtype=int32)

    In [2]: ph.ubnd_counts                                                                                                                                                                              
    Out[2]: array([53318,  2528,  6854])


::

    O[blyth@localhost ~]$ cat /tmp/blyth/opticks/GeoChain/body_phys/CSGFoundry/meshname.txt 
    PMTSim_inner1_solid_I
    PMTSim_inner2_solid_1_9
    PMTSim_body_solid_1_9

    O[blyth@localhost ~]$ cat /tmp/blyth/opticks/GeoChain/body_phys/CSGFoundry/bnd.txt
    Pyrex///Pyrex
    Pyrex/PMTSim_photocathode_logsurf2/PMTSim_photocathode_logsurf1/Vacuum
    Pyrex/PMTSim_mirror_logsurf2/PMTSim_mirror_logsurf1/Vacuum



Need G4 ground truth to compare against for volumes as well as solids
------------------------------------------------------------------------

::

   g4-
   g4-cls G4VPhysicalVolume    # there are no DistanceToIn DistanceToOut methods
   g4-cls G4RayTracer
   g4-cls G4TheRayTracer       # this must do something like what I need 
   
   G4TheRayTracer::CreateBitMap

   G4RayShooter::Shoot  
       adds geantino primary vertex to G4Event 

   g4-cls G4RTSteppingAction
       stops and kills track depending on transparency settings

   g4-cls G4RTTrackingAction
       does little


Possible nudge issue with body_phys
-------------------------------------

* looks like an equatorial sombrero : this was fixed following avoiding of stomping on ellipsoid scaling transform
* now suspect the dynode solids ?

::

   gc
   ./run.sh   # volume test with body_phys 

   cx
   om
   ./cxr_geochain.sh   # with body_phys    




Possible cause of why --skipsolidname not working
-----------------------------------------------------

* skip logic only in GInstancer::labelRepeats_r and not in GInstancer::labelGlobals_r


* moved skipping logic in GInstancer into GInstancer::visitNode so can 
  call from labelRepeat_r or labelGlobals_r however the notes in 
  why cannot do global level solid skips at such a late stage seem to 
  suggest its not worth pursuing. 

* BUT considering alternatives GInstancer seems like the natural place to skip
  because the volumes are already partitioned there 

* instead look at X4PhysicalVolume::convertStructure that grabs the 
  GMesh created in X4PhysicalVolume::convertSolid 

  * but the natural way to do things there is to set a flag on the GMesh 
    which is used from the GNode, which boils down to the same skip in 
    GInstancer... so need to face whats going wrong with global skips 
   



