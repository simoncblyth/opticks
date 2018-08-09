ab-blib : boundary surface mismatch
=====================================


FIXES : by changing A to check for parent lv skin surfaces
------------------------------------------------------------

Reworking both A and B succeeds to get them to agree. 
As well as fixing some name resolution bugs had to make
a significant change to A : checking for skin surfaces *sks_p*
of the parent LV *lv_p*. 


ab-bnd
-------

::

    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/bin/ab/ab-bnd.py
     ia.min() 17 ia.max() 126  len(np.unique(ia)) 100  
     ib.min() 17 ib.max() 126  len(np.unique(ib)) 100  
    [2018-08-09 17:21:59,037] p11083 {/tmp/blyth/opticks/bin/ab/ab-bnd.py:25} INFO -  part.bnd diff :  0/11984 
    []
    np.all( ia[:100] == ib[:100] )  True
    np.all( ia[:1000] == ib[:1000] )  True
    np.all( ia[:2000] == ib[:2000] )  True
    np.all( ia[:3000] == ib[:3000] )  True
    np.all( ia[:4000] == ib[:4000] )  True
    np.all( ia[:5000] == ib[:5000] )  True
    np.all( ia[:6000] == ib[:6000] )  True
    np.all( ia[:7000] == ib[:7000] )  True
    np.all( ia[:8000] == ib[:8000] )  True
    np.all( ia[:9000] == ib[:9000] )  True
    np.all( ia == ib )  True



G4OpBoundaryProcess surface resolution
----------------------------------------

g4-cls G4OpBoundaryProcess::

     165 G4VParticleChange*
     166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     167 {
     ...
     306         theModel = glisur;
     307         theFinish = polished;
     308 
     309         G4SurfaceType type = dielectric_dielectric;
     310 
     311         Rindex = NULL;
     312         OpticalSurface = NULL;
     313 
     314         G4LogicalSurface* Surface = NULL;
     315 
     316         Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
     317 
     318         if (Surface == NULL){
     319           G4bool enteredDaughter= (thePostPV->GetMotherLogical() ==
     320                                    thePrePV ->GetLogicalVolume());
     ///
     ///          stepping between parent PV and child PV
     ///
     321       if(enteredDaughter){
     322         Surface =
     323               G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     324         if(Surface == NULL)
     325           Surface =
     326                 G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     327       }
     328       else {
     329         Surface =
     330               G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
     331         if(Surface == NULL)
     332           Surface =
     333                 G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
     334       }
     335     }
     336 
     337     if (Surface) OpticalSurface =
     338            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
     339 
     340     if (OpticalSurface) {
     341 
     342            type      = OpticalSurface->GetType();
     343        theModel  = OpticalSurface->GetModel();
     344        theFinish = OpticalSurface->GetFinish();
     345 
     346        aMaterialPropertiesTable = OpticalSurface->
     347                     GetMaterialPropertiesTable();
     348 


Notice:

1. border surface has 1st try and has directionality : this makes sense 
2. when stepping from parent PV to child PV : child and parent lv both tried for skin surface in child then parent order (inner first)
3. when stepping from child PV to parent PV : child and parent lv both tried for skin surface in parent then child order (outer first)

* a possible skinsurface on the volume where the photon is headed gets 1st priority    
* then try for another possible skinsurface on the lv where the photon is leaving 


Not sure that opticks boundaries can model this ... Looked into this before somewhere 

* actually Opticks could check osurf then isurf on GPU : but need to be motivated by a real problem to pursue that 
* suspect the use of adjacent doubleskin surfaces in parent-child is rare 
* skin surfaces make kinda sense for a leaf node but a border surface is much clearer and cleaner


Boundary Model Mapping Issue  
------------------------------

* the real answer will come with detailed vs G4 comparisons : but for now
  need to match boundaries 


::

    epsilon:issues blyth$ ab-blib-smry 
    A
     nbnd 123 nmat  38 nsur  48 
      0 : Vacuum///Vacuum 
      1 : Vacuum///Rock 
      2 : Rock///Air 
      3 : Air/NearPoolCoverSurface//PPE 
      4 : Air///Aluminium 
    118 : OwsWater/VertiCableTraySurface//UnstStainlessSteel 
    119 : OwsWater/NearOutInPiperSurface//PVC 
    120 : OwsWater/NearOutOutPiperSurface//PVC 
    121 : DeadWater/LegInDeadTubSurface//ADTableStainlessSteel 
    122 : Rock///RadRock 
    B
     nbnd 127 nmat  38 nsur  48 
      0 : Vacuum///Vacuum 
      1 : Vacuum///Rock 
      2 : Rock///Air 
      3 : Air/NearPoolCoverSurface/NearPoolCoverSurface/PPE 
      4 : Air///Aluminium 
    122 : UnstStainlessSteel/VertiCableTraySurface/VertiCableTraySurface/BPE 
    123 : OwsWater/NearOutInPiperSurface/NearOutInPiperSurface/PVC 
    124 : OwsWater/NearOutOutPiperSurface/NearOutOutPiperSurface/PVC 
    125 : DeadWater/LegInDeadTubSurface/LegInDeadTubSurface/ADTableStainlessSteel 
    126 : Rock///RadRock 
    epsilon:issues blyth$ 


B :  X4PhysicalVolume::findSurface is attempting to mimic G4OpBoundaryProcess (extent to be verified)
----------------------------------------------------------------------------------------------------------------

::

    325 /**
    326 X4PhysicalVolume::findSurface
    327 ------------------------------
    328 
    329 1. look for a border surface from PV_a to PV_b (do not look for the opposite direction)
    330 2. if no border surface look for a logical skin surface with the lv of the first PV_a otherwise the lv of PV_b 
    331    (or vv when first_priority is false) 
    332 
    333 **/
    334 
    335 G4LogicalSurface* X4PhysicalVolume::findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority )
    336 {
    337      G4LogicalSurface* surf = G4LogicalBorderSurface::GetSurface(a, b) ;
    338 
    339      const G4VPhysicalVolume* const first  = first_priority ? a : b ;
    340      const G4VPhysicalVolume* const second = first_priority ? b : a ;
    341 
    342      if(surf == NULL)
    343          surf = G4LogicalSkinSurface::GetSurface(first ? first->GetLogicalVolume() : NULL );
    344 
    345      if(surf == NULL)
    346          surf = G4LogicalSkinSurface::GetSurface(second ? second->GetLogicalVolume() : NULL );
    347 
    348      return surf ;
    349 }


    543 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
    544 {
    545      const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    546      const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
    547 
    548      const G4Material* const imat = lv->GetMaterial() ;
    549      const G4Material* const omat = lv_p ? lv_p->GetMaterial() : imat ;  // top omat -> imat 
    550 
    551      bool first_priority = true ;
    552      const G4LogicalSurface* const isur = findSurface( pv  , pv_p , first_priority );
    553      const G4LogicalSurface* const osur = findSurface( pv_p, pv   , first_priority );
    554      // doubtful of findSurface priority with double skin surfaces, see g4op-
    555 
    556      unsigned boundary = m_blib->addBoundary(
    557                                                 X4::BaseName(omat),
    558                                                 X4::BaseName(osur),
    559                                                 X4::BaseName(isur),
    560                                                 X4::BaseName(imat)
    561                                             );
    562      return boundary ;
    563 }





A : AssimpGGeo::convertStructureVisit reworked
-------------------------------------------------

::

    0966     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     967     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     968     GSkinSurface*   sks = gg->findSkinSurface(lv);
     969     GSkinSurface*   sks_p = gg->findSkinSurface(lv_p);
     970     // dont like sks_p : but it seems to correspond with G4OpBoundary surface resolution see notes/issues/ab-blib.rst
     971 
     972     unsigned int nsurf = 0 ;
     973     if(sks) nsurf++ ;
     974     if(ibs) nsurf++ ;
     975     if(obs) nsurf++ ;
     976     assert(nsurf == 0 || nsurf == 1 || nsurf == 2);
     977 
     978     // see notes/issues/ab-blib.rst 
     979 
     980     if(obs)
     981     {
     982         osurf = obs ;
     983         isurf = NULL ;
     984         m_outborder_surface++ ;
     985     }   
     986     else if(ibs)
     987     {
     988         osurf = NULL ;
     989         isurf = ibs ; 
     990         m_inborder_surface++ ;
     991     }   
     992     else if(sks && !sks_p)
     993     {
     994         osurf = sks ;
     995         isurf = sks ;
     996         m_skin_surface++ ;
     997     }   
     998     else if(!sks && sks_p )
     999     {
    1000         osurf = sks_p ;
    1001         isurf = sks_p ;
    1002         m_skin_surface++ ;
    1003     }   
    1004     else if(sks && sks_p ) // doubleskin : not yet seen in wild 
    1005     {
    1006         assert(0);
    1007         bool swap = false ;    // unsure ... needs some testing vs G4
    1008         osurf = swap ? sks_p : sks   ;
    1009         isurf = swap ? sks   : sks_p ;
    1010         m_doubleskin_surface++ ;
    1011     }
    1012     else
    1013     {
    1014         m_no_surface++ ;
    1015     }



A  : AssimpGGeo::convertStructureVisit
-------------------------------------------


::


    0949     GBorderSurface* obs = gg->findBorderSurface(pv_p, pv);  // outer surface (parent->self) 
     950     GBorderSurface* ibs = gg->findBorderSurface(pv, pv_p);  // inner surface (self->parent) 
     951     GSkinSurface*   sks = gg->findSkinSurface(lv);
     952 

     976     GPropertyMap<float>* isurf  = NULL ;
     977     GPropertyMap<float>* osurf  = NULL ;
     980 
     981     if(sks)
     982     {
     983         osurf = sks ;
                 isurf = sks ;   // try this to align the algos
     990     }
     991     else if(obs)
     992     {
     993         osurf = obs ;
     999     }
    1000     else if(ibs)
    1001     {
    1002         isurf = ibs ;
    1008     }

    1025     // boundary identification via 4-uint 
    1026     boundary = blib->addBoundary(
    1027                                   mt_p->getShortName(),
    1028                                   osurf ? osurf->getShortName() : NULL ,
    1029                                   isurf ? isurf->getShortName() : NULL ,
    1030                                   mt->getShortName()
    1031                                   );
    1032 




After adding the isurf for sks to B, get closer : but a few discreps remain::

    epsilon:ab-blib-diff blyth$ ab-blib-diff2
    diff -y /tmp/blyth/opticks/bin/ab/ab-blib-diff2/a2.txt /tmp/blyth/opticks/bin/ab/ab-blib-diff2/b2.txt --width 180
    MineralOil///Pyrex                                                                  MineralOil///Pyrex
    Pyrex///Vacuum                                                                      Pyrex///Vacuum
    Vacuum/lvPmtHemiCathodeSensorSurface/lvPmtHemiCathodeSensorSurface/Bialkali       | Vacuum///Bialkali
    Vacuum///OpaqueVacuum                                                               Vacuum///OpaqueVacuum
    MineralOil///UnstStainlessSteel                                                     MineralOil///UnstStainlessSteel
    MineralOil///Vacuum                                                                 MineralOil///Vacuum
    Vacuum///Pyrex                                                                      Vacuum///Pyrex
    Vacuum/lvHeadonPmtCathodeSensorSurface/lvHeadonPmtCathodeSensorSurface/Bialkali   <
    Vacuum///PVC                                                                        Vacuum///PVC
    MineralOil///StainlessSteel                                                         MineralOil///StainlessSteel
    MineralOil/RSOilSurface/RSOilSurface/Acrylic                                        MineralOil/RSOilSurface/RSOilSurface/Acrylic
    ...
    Nitrogen///LiquidScintillator                                                       Nitrogen///LiquidScintillator
    IwsWater/AdCableTraySurface/AdCableTraySurface/UnstStainlessSteel                   IwsWater/AdCableTraySurface/AdCableTraySurface/UnstStainlessSteel
    UnstStainlessSteel///BPE                                                          | UnstStainlessSteel/AdCableTraySurface/AdCableTraySurface/BPE
    Water///Nitrogen                                                                    Water///Nitrogen
    Nitrogen///MineralOil                                                               Nitrogen///MineralOil
    ...
    IwsWater/SlopeRib5Surface/SlopeRib5Surface/ADTableStainlessSteel                    IwsWater/SlopeRib5Surface/SlopeRib5Surface/ADTableStainlessSteel
    IwsWater/ADVertiCableTraySurface/ADVertiCableTraySurface/UnstStainlessSteel         IwsWater/ADVertiCableTraySurface/ADVertiCableTraySurface/UnstStainlessSteel
                                                                                      > UnstStainlessSteel/ADVertiCableTraySurface/ADVertiCableTraySurface/BPE
    IwsWater/ShortParCableTraySurface/ShortParCableTraySurface/UnstStainlessSteel       IwsWater/ShortParCableTraySurface/ShortParCableTraySurface/UnstStainlessSteel
                                                                                      > UnstStainlessSteel/ShortParCableTraySurface/ShortParCableTraySurface/BPE
    IwsWater/NearInnInPiperSurface/NearInnInPiperSurface/PVC                            IwsWater/NearInnInPiperSurface/NearInnInPiperSurface/PVC
    ...
    OwsWater/UnistrutRib9Surface/UnistrutRib9Surface/UnstStainlessSteel                 OwsWater/UnistrutRib9Surface/UnistrutRib9Surface/UnstStainlessSteel
    OwsWater/TopShortCableTraySurface/TopShortCableTraySurface/UnstStainlessSteel       OwsWater/TopShortCableTraySurface/TopShortCableTraySurface/UnstStainlessSteel
                                                                                      > UnstStainlessSteel/TopShortCableTraySurface/TopShortCableTraySurface/BPE
    OwsWater/TopCornerCableTraySurface/TopCornerCableTraySurface/UnstStainlessSteel     OwsWater/TopCornerCableTraySurface/TopCornerCableTraySurface/UnstStainlessSteel
                                                                                      > UnstStainlessSteel/TopCornerCableTraySurface/TopCornerCableTraySurface/BPE
    OwsWater/VertiCableTraySurface/VertiCableTraySurface/UnstStainlessSteel             OwsWater/VertiCableTraySurface/VertiCableTraySurface/UnstStainlessSteel
                                                                                      > UnstStainlessSteel/VertiCableTraySurface/VertiCableTraySurface/BPE
    OwsWater/NearOutInPiperSurface/NearOutInPiperSurface/PVC                            OwsWater/NearOutInPiperSurface/NearOutInPiperSurface/PVC


Issue 1 : B fails to see the SensorSurfaces that A does : B-1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SensorSurfaces are artifical additions to the Opticks model, so 
because B is following G4 way of finding surfaces it dont find them.
A (AssimpGGeo) follows a more Opticks approach and so sees the sensor surfaces ?  

::

     A sees 

     Vacuum/lvPmtHemiCathodeSensorSurface/lvPmtHemiCathodeSensorSurface/Bialkali
     Vacuum/lvHeadonPmtCathodeSensorSurface/lvHeadonPmtCathodeSensorSurface/Bialkali

     B sees 

     Vacuum///Bialkali


Issue 2 : A fails to see surface sandwich filling UnstStainlessSteel///BPE : B+5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    A sees one surface-less boundary 

    UnstStainlessSteel///BPE

    B sees six different sandwiches

    UnstStainlessSteel/AdCableTraySurface/AdCableTraySurface/BPE
    UnstStainlessSteel/ADVertiCableTraySurface/ADVertiCableTraySurface/BPE
    UnstStainlessSteel/ShortParCableTraySurface/ShortParCableTraySurface/BPE
    UnstStainlessSteel/TopShortCableTraySurface/TopShortCableTraySurface/BPE
    UnstStainlessSteel/TopCornerCableTraySurface/TopCornerCableTraySurface/BPE
    UnstStainlessSteel/VertiCableTraySurface/VertiCableTraySurface/BPE

    ### all surfaces that A fails to find are *CableTraySurface 
    ### which are skin surfaces with LV listed below according to B



A/GNodeLib/LVNames.txt::

    12180 __dd__Geometry__PoolDetails__lvTopCornerCableTray0xce56ff8
    12181 __dd__Geometry__PoolDetails__lvTopCornerCable0xce57340
    12182 __dd__Geometry__PoolDetails__lvVertiCableTray0xc0e08a0
    12183 __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    12184 __dd__Geometry__PoolDetails__lvVertiCableTray0xc0e08a0
    12185 __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    12186 __dd__Geometry__PoolDetails__lvVertiCableTray0xc0e08a0
    12187 __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    12188 __dd__Geometry__PoolDetails__lvVertiCableTray0xc0e08a0
    12189 __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    12190 __dd__Geometry__PoolDetails__lvTopShortCableTray0xce58200
    12191 __dd__Geometry__PoolDetails__lvTopShortCable0xce57be8

B/GNodeLib/LVNames.txt::

    12180 /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
    12181 /dd/Geometry/PoolDetails/lvTopCornerCable0xce57340
    12182 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12183 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12184 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12185 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12186 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12187 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12188 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12189 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12190 /dd/Geometry/PoolDetails/lvTopShortCableTray0xce58200
    12191 /dd/Geometry/PoolDetails/lvTopShortCable0xce57be8





problem pair dumping

B::

    :set nowrap

    2018-08-09 14:38:13.778 INFO  [11353628] [GSurfaceLib::dumpSkinSurface@1286] dumpSkinSurface
     SS    0 :                     NearPoolCoverSurface : /dd/Geometry/PoolDetails/lvNearTopCover0xc137060
     SS    1 :                             RSOilSurface : /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0
     SS    2 :            *           AdCableTraySurface : /dd/Geometry/AdDetails/lvAdVertiCableTray0xc3a27f0
     SS    3 :                      PmtMtTopRingSurface : /dd/Geometry/PMT/lvPmtTopRing0xc3486f0
     SS    4 :                     PmtMtBaseRingSurface : /dd/Geometry/PMT/lvPmtBaseRing0xc00f400
     SS    5 :                         PmtMtRib1Surface : /dd/Geometry/PMT/lvMountRib10xc3a4cb0
     SS    6 :                         PmtMtRib2Surface : /dd/Geometry/PMT/lvMountRib20xc012500
     SS    7 :                         PmtMtRib3Surface : /dd/Geometry/PMT/lvMountRib30xc00d350
     SS    8 :                       LegInIWSTubSurface : /dd/Geometry/PoolDetails/lvLegInIWSTub0xc400e40
     SS    9 :                        TablePanelSurface : /dd/Geometry/PoolDetails/lvTablePanel0xc0101d8
     SS   10 :                       SupportRib1Surface : /dd/Geometry/PoolDetails/lvSupportRib10xc0d8868
     SS   11 :                       SupportRib5Surface : /dd/Geometry/PoolDetails/lvSupportRib50xc0d8bb8
     SS   12 :                         SlopeRib1Surface : /dd/Geometry/PoolDetails/lvSlopeRib10xc0d8b50
     SS   13 :                         SlopeRib5Surface : /dd/Geometry/PoolDetails/lvSlopeRib50xc0d8db0
     SS   14 :            *      ADVertiCableTraySurface : /dd/Geometry/PoolDetails/lvInnVertiCableTray0xbf28e40
     SS   15 :            *     ShortParCableTraySurface : /dd/Geometry/PoolDetails/lvInnShortParCableTray0xc95a730
     SS   16 :                    NearInnInPiperSurface : /dd/Geometry/PoolDetails/lvInnInWaterPipeNearTub0xbf29660
     SS   17 :                   NearInnOutPiperSurface : /dd/Geometry/PoolDetails/lvInnOutWaterPipeNearTub0xc0d7c30
     SS   18 :                       LegInOWSTubSurface : /dd/Geometry/PoolDetails/lvLegInOWSTub0xcced348
     SS   19 :                      UnistrutRib6Surface : /dd/Geometry/PoolDetails/lvShortParRib10xcd55e48
     SS   20 :                      UnistrutRib7Surface : /dd/Geometry/PoolDetails/lvShortParRib20xcd56b40
     SS   21 :                      UnistrutRib3Surface : /dd/Geometry/PoolDetails/lvBotVertiRib0xbf63800
     SS   22 :                      UnistrutRib5Surface : /dd/Geometry/PoolDetails/lvCrossRib0xcd570b8
     SS   23 :                      UnistrutRib4Surface : /dd/Geometry/PoolDetails/lvSidVertiRib0xc5e6fa0
     SS   24 :                      UnistrutRib1Surface : /dd/Geometry/PoolDetails/lvLongParRib10xc3b3eb8
     SS   25 :                      UnistrutRib2Surface : /dd/Geometry/PoolDetails/lvLongParRib20xc3b4910
     SS   26 :                      UnistrutRib8Surface : /dd/Geometry/PoolDetails/lvCornerParRib10xc0e2430
     SS   27 :                      UnistrutRib9Surface : /dd/Geometry/PoolDetails/lvCornerParRib20xc0f2040
     SS   28 :            *     TopShortCableTraySurface : /dd/Geometry/PoolDetails/lvTopShortCableTray0xce58200
     SS   29 :            *    TopCornerCableTraySurface : /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
     SS   30 :            *        VertiCableTraySurface : /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
     SS   31 :                    NearOutInPiperSurface : /dd/Geometry/PoolDetails/lvOutInWaterPipeNearTub0xce594c0
     SS   32 :                   NearOutOutPiperSurface : /dd/Geometry/PoolDetails/lvOutOutWaterPipeNearTub0xce58ca0
     SS   33 :                      LegInDeadTubSurface : /dd/Geometry/PoolDetails/lvLegInDeadTub0xce5bea8
     SS   34 :          lvHeadonPmtCathodeSensorSurface : lvHeadonPmtCathode0xc2c8d98
     SS   35 :            lvPmtHemiCathodeSensorSurface : lvPmtHemiCathode0xc2cdca0



    2018-08-09 14:28:16.585 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 4775 isur_ 0x7fb4160e97a0 osur_ 0x7fb4160e97a0 _lv lvAdVertiCable0xc2d1f60 _lv_p lvAdVertiCableTray0xc3a27f0
    2018-08-09 14:28:16.692 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 6435 isur_ 0x7fb4160e97a0 osur_ 0x7fb4160e97a0 _lv lvAdVertiCable0xc2d1f60 _lv_p lvAdVertiCableTray0xc3a27f0
    2018-08-09 14:28:16.818 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 8569 isur_ 0x7fb4160f1670 osur_ 0x7fb4160f1670 _lv lvInnVertiCable0xbf28d50 _lv_p lvInnVertiCableTray0xbf28e40
    2018-08-09 14:28:16.819 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 8571 isur_ 0x7fb4160f1670 osur_ 0x7fb4160f1670 _lv lvInnVertiCable0xbf28d50 _lv_p lvInnVertiCableTray0xbf28e40
    2018-08-09 14:28:16.819 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 8573 isur_ 0x7fb4160f2b20 osur_ 0x7fb4160f2b20 _lv lvInnShortParCable0xbf6f630 _lv_p lvInnShortParCableTray0xc95a730
    2018-08-09 14:28:16.819 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 8575 isur_ 0x7fb4160f2b20 osur_ 0x7fb4160f2b20 _lv lvInnShortParCable0xbf6f630 _lv_p lvInnShortParCableTray0xc95a730
    2018-08-09 14:28:17.019 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12176 isur_ 0x7fb416700940 osur_ 0x7fb416700940 _lv lvTopShortCable0xce57be8 _lv_p lvTopShortCableTray0xce58200
    2018-08-09 14:28:17.019 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12178 isur_ 0x7fb416701300 osur_ 0x7fb416701300 _lv lvTopCornerCable0xce57340 _lv_p lvTopCornerCableTray0xce56ff8
    2018-08-09 14:28:17.019 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12180 isur_ 0x7fb416701300 osur_ 0x7fb416701300 _lv lvTopCornerCable0xce57340 _lv_p lvTopCornerCableTray0xce56ff8
    2018-08-09 14:28:17.019 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12182 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.019 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12184 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.020 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12186 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.020 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12188 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.020 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12190 isur_ 0x7fb416700940 osur_ 0x7fb416700940 _lv lvTopShortCable0xce57be8 _lv_p lvTopShortCableTray0xce58200
    2018-08-09 14:28:17.020 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12192 isur_ 0x7fb416701300 osur_ 0x7fb416701300 _lv lvTopCornerCable0xce57340 _lv_p lvTopCornerCableTray0xce56ff8
    2018-08-09 14:28:17.020 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12194 isur_ 0x7fb416701300 osur_ 0x7fb416701300 _lv lvTopCornerCable0xce57340 _lv_p lvTopCornerCableTray0xce56ff8
    2018-08-09 14:28:17.021 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12196 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.021 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12198 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.021 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12200 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0
    2018-08-09 14:28:17.021 ERROR [11346040] [X4PhysicalVolume::addBoundary@598]  problem_pair  node_count 12202 isur_ 0x7fb416700e90 osur_ 0x7fb416700e90 _lv lvVertiCable0xce570c8 _lv_p lvVertiCableTray0xc0e08a0

    ##### what happed to the Tray LV ? 
 

B/GNodeLib/LVNames.txt (with World deleted so the 1-based vim line numbers match 0-based node index)::

    12174 /dd/Geometry/PoolDetails/lvCrossRib0xcd570b8
    12175 /dd/Geometry/PoolDetails/lvTopShortCableTray0xce58200
    12176 /dd/Geometry/PoolDetails/lvTopShortCable0xce57be8
    12177 /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
    12178 /dd/Geometry/PoolDetails/lvTopCornerCable0xce57340
    12179 /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
    12180 /dd/Geometry/PoolDetails/lvTopCornerCable0xce57340
    12181 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12182 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12183 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12184 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12185 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12186 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12187 /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0
    12188 /dd/Geometry/PoolDetails/lvVertiCable0xce570c8
    12189 /dd/Geometry/PoolDetails/lvTopShortCableTray0xce58200
    12190 /dd/Geometry/PoolDetails/lvTopShortCable0xce57be8
    12191 /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
    12192 /dd/Geometry/PoolDetails/lvTopCornerCable0xce57340
    12193 /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8
    12194 /dd/Geometry/PoolDetails/lvTopCornerCable0xce57340


A::

    2018-08-09 14:47:58.176 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  4775 lv __dd__Geometry__AdDetails__lvAdVertiCable0xc2d1f60
    2018-08-09 14:47:58.220 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  6435 lv __dd__Geometry__AdDetails__lvAdVertiCable0xc2d1f60
    2018-08-09 14:47:58.294 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  8569 lv __dd__Geometry__PoolDetails__lvInnVertiCable0xbf28d50
    2018-08-09 14:47:58.294 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  8571 lv __dd__Geometry__PoolDetails__lvInnVertiCable0xbf28d50
    2018-08-09 14:47:58.295 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  8573 lv __dd__Geometry__PoolDetails__lvInnShortParCable0xbf6f630
    2018-08-09 14:47:58.295 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex  8575 lv __dd__Geometry__PoolDetails__lvInnShortParCable0xbf6f630
    2018-08-09 14:47:58.451 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12176 lv __dd__Geometry__PoolDetails__lvTopShortCable0xce57be8
    2018-08-09 14:47:58.451 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12178 lv __dd__Geometry__PoolDetails__lvTopCornerCable0xce57340
    2018-08-09 14:47:58.451 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12180 lv __dd__Geometry__PoolDetails__lvTopCornerCable0xce57340
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12182 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12184 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12186 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12188 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12190 lv __dd__Geometry__PoolDetails__lvTopShortCable0xce57be8
    2018-08-09 14:47:58.452 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12192 lv __dd__Geometry__PoolDetails__lvTopCornerCable0xce57340
    2018-08-09 14:47:58.453 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12194 lv __dd__Geometry__PoolDetails__lvTopCornerCable0xce57340
    2018-08-09 14:47:58.453 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12196 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.453 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12198 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.453 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12200 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8
    2018-08-09 14:47:58.453 ERROR [11359450] [*AssimpGGeo::convertStructureVisit@954] problem_pair nodeIndex 12202 lv __dd__Geometry__PoolDetails__lvVertiCable0xce570c8



GDML cables in trays
-----------------------

::


    01678     <box lunit="mm" name="ad_verti_cable0xc182f50" x="55" y="20" z="5000"/>
     1679     <box lunit="mm" name="ad-verti_box0xc3d7668" x="60" y="40" z="5000"/>
     1680     <box lunit="mm" name="ad-verti_box_sub0xc580578" x="55" y="27.5" z="5010"/>
     1681     <subtraction name="ad-verti_cable_tray0xbf53868">
     1682       <first ref="ad-verti_box0xc3d7668"/>
     1683       <second ref="ad-verti_box_sub0xc580578"/>
     1684       <position name="ad-verti_cable_tray0xbf53868_pos" unit="mm" x="0" y="-16.25" z="0"/>
     1685     </subtraction>


     7638     <volume name="/dd/Geometry/AdDetails/lvAdVertiCable0xc2d1f60">
     7639       <materialref ref="/dd/Materials/BPE0xc0ad360"/>
     7640       <solidref ref="ad_verti_cable0xc182f50"/>
     7641     </volume>
     7642     <volume name="/dd/Geometry/AdDetails/lvAdVertiCableTray0xc3a27f0">
     7643       <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
     7644       <solidref ref="ad-verti_cable_tray0xbf53868"/>
     7645       <physvol name="/dd/Geometry/AdDetails/lvAdVertiCableTray#pvAdVertiCable0xc3a5a90">
     7646         <volumeref ref="/dd/Geometry/AdDetails/lvAdVertiCable0xc2d1f60"/>
     7647         <position name="/dd/Geometry/AdDetails/lvAdVertiCableTray#pvAdVertiCable0xc3a5a90_pos" unit="mm" x="0" y="7.5" z="0"/>
     7648       </physvol>
     7649     </volume>


    01678     <box lunit="mm" name="ad_verti_cable0xc182f50" x="55" y="20" z="5000"/>
     1679     <box lunit="mm" name="ad-verti_box0xc3d7668" x="60" y="40" z="5000"/>
     1680     <box lunit="mm" name="ad-verti_box_sub0xc580578" x="55" y="27.5" z="5010"/>
     1681     <subtraction name="ad-verti_cable_tray0xbf53868">
     1682       <first ref="ad-verti_box0xc3d7668"/>
     1683       <second ref="ad-verti_box_sub0xc580578"/>
     1684       <position name="ad-verti_cable_tray0xbf53868_pos" unit="mm" x="0" y="-16.25" z="0"/>
     1685     </subtraction>


    16126     <volume name="/dd/Geometry/PoolDetails/lvVertiCable0xce570c8">
    16127       <materialref ref="/dd/Materials/BPE0xc0ad360"/>
    16128       <solidref ref="verti_cable0xc0e0d70"/>
    16129     </volume>
    16130     <volume name="/dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0">
    16131       <materialref ref="/dd/Materials/UnstStainlessSteel0xc5c11e8"/>
    16132       <solidref ref="verti_cable_tray0xce58a50"/>
    16133       <physvol name="/dd/Geometry/PoolDetails/lvVertiCableTray#pvVertiCable0xbf5e7f0">
    16134         <volumeref ref="/dd/Geometry/PoolDetails/lvVertiCable0xce570c8"/>
    16135         <position name="/dd/Geometry/PoolDetails/lvVertiCableTray#pvVertiCable0xbf5e7f0_pos" unit="mm" x="0" y="7.5" z="0"/>
    16136       </physvol>
    16137     </volume>



G4DAE the skinsurface logical volume for the trays is not a leaf node::

    153269       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib9Surface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__UnistrutRib9Surface">
    153270         <volumeref ref="__dd__Geometry__PoolDetails__lvCornerParRib20xc0f2040"/>
    153271       </skinsurface>
    153272       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__TopShortCableTraySurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__TopShortCableTraySurface">
    153273         <volumeref ref="__dd__Geometry__PoolDetails__lvTopShortCableTray0xce58200"/>
    153274       </skinsurface>
    153275       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__TopCornerCableTraySurface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__TopCornerCableTraySurface">
    153276         <volumeref ref="__dd__Geometry__PoolDetails__lvTopCornerCableTray0xce56ff8"/>
    153277       </skinsurface>
    153278       <skinsurface name="__dd__Geometry__PoolDetails__PoolSurfacesAll__VertiCableTraySurface" surfaceproperty="__dd__Geometry__PoolDetails__PoolSurfacesAll__VertiCableTraySurface">
    153279         <volumeref ref="__dd__Geometry__PoolDetails__lvVertiCableTray0xc0e08a0"/>
    153280       </skinsurface>





