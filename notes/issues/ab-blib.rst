ab-blib : boundary surface mismatch
=====================================

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



