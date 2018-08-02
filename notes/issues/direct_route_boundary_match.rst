direct_route_boundary_match
=============================


ab-blib 
---------

Observations: 

1. same number of materials
2. two extra surfaces in B ( lvPmtHemiCathodeSensorSurface, lvHeadonPmtCathodeSensorSurface )

   * that was with 103 (for the reference), get the same with 1 for the reference
     BUT : the order is different 

   * these are artificial additions.. for model matching 
   * what was wrong with the old one are comparing against ?
   * forget the details, but twas something to do with it being easier to detect a 
     hit on a surface : in the Opticks surface model, so I added surfaces to the cathodes  





::

     534 /**
     535 AssimpGGeo::convertSensors
     536 ---------------------------
     537 
     538 Opticks is a surface based simulation, as opposed to 
     539 Geant4 which is CSG volume based. In Geant4 hits are formed 
     540 on stepping into volumes with associated SensDet.
     541 The Opticks equivalent is intersecting with a "SensorSurface", 
     542 which are fabricated by AssimpGGeo::convertSensors.
     543 
     544 **/
     545 





3. B is sometimes duplicating isur/osur but A is not 

   * think this was a fix to better translate the Geant4 meaning of border (with directionality)
     vs skin surfaces (without directionality)  


A
nbnd 122 nmat  38 nsur  46 
  0 : Vacuum///Vacuum 
  1 : Vacuum///Rock 
  2 : Rock///Air 
  3 : Air/NearPoolCoverSurface//PPE 
  4 : Air///Aluminium 
  5 : Aluminium///Foam 
...
120 : DeadWater/LegInDeadTubSurface//ADTableStainlessSteel 
121 : Rock///RadRock 

B
 nbnd 128 nmat  38 nsur  48 
  0 : Vacuum///Vacuum 
  1 : Vacuum///Rock 
  2 : Rock///Air 
  3 : Air/NearPoolCoverSurface/NearPoolCoverSurface/PPE 






::

    410 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
    411 {
    412      const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
    413      const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
    414 
    415      const G4Material* const imat = lv->GetMaterial() ;
    416      const G4Material* const omat = lv_p ? lv_p->GetMaterial() : imat ;  // top omat -> imat 
    417 
    418      bool first_priority = true ;
    419      const G4LogicalSurface* const isur = findSurface( pv  , pv_p , first_priority );
    420      const G4LogicalSurface* const osur = findSurface( pv_p, pv   , first_priority );
    421      // doubtful of findSurface priority with double skin surfaces, see g4op-
    422 
    423      unsigned boundary = m_blib->addBoundary(
    424                                                 X4::BaseName(omat),
    425                                                 X4::BaseName(osur),
    426                                                 X4::BaseName(isur),
    427                                                 X4::BaseName(imat)
    428                                             );
    429      return boundary ;
    430 }

    330 G4LogicalSurface* X4PhysicalVolume::findSurface( const G4VPhysicalVolume* const a, const G4VPhysicalVolume* const b, bool first_priority )
    331 {
    332      G4LogicalSurface* surf = G4LogicalBorderSurface::GetSurface(a, b) ;
    333 
    334      const G4VPhysicalVolume* const first  = first_priority ? a : b ;
    335      const G4VPhysicalVolume* const second = first_priority ? b : a ;
    336 
    337      if(surf == NULL)
    338          surf = G4LogicalSkinSurface::GetSurface(first ? first->GetLogicalVolume() : NULL );
    339 
    340      if(surf == NULL)
    341          surf = G4LogicalSkinSurface::GetSurface(second ? second->GetLogicalVolume() : NULL );
    342 
    343      return surf ;
    344 }





Why did the old 103 miss the sensors ?
-----------------------------------------


Comparison with old 1 (not 103) and direct gives same surface count, but order differs
-----------------------------------------------------------------------------------------

::

    epsilon:opticksdata blyth$ find . -name order.json
    ./export/DayaBay/GMaterialLib/order.json
    ./export/DayaBay/GSurfaceLib/order.json




    127 : Rock///RadRock 
    A
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GSurfaceLib
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GSurfaceLib/GSurfaceLibOptical.npy : (48, 4) 
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GSurfaceLib/GSurfaceLib.npy : (48, 2, 39, 4) 
    B
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GSurfaceLib
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GSurfaceLib/GSurfaceLibOptical.npy : (48, 4) 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GSurfaceLib/GSurfaceLib.npy : (48, 2, 39, 4) 

    NearPoolCoverSurface					      <
    NearDeadLinerSurface						NearDeadLinerSurface
    NearOWSLinerSurface						NearOWSLinerSurface
    NearIWSCurtainSurface						NearIWSCurtainSurface
    SSTWaterSurfaceNear1						SSTWaterSurfaceNear1
    SSTOilSurface							SSTOilSurface
                                      >	ESRAirSurfaceTop
                                      >	ESRAirSurfaceBot
                                      >	SSTWaterSurfaceNear2
                                      >	NearPoolCoverSurface
    lvPmtHemiCathodeSensorSurface					lvPmtHemiCathodeSensorSurface
    lvHeadonPmtCathodeSensorSurface					lvHeadonPmtCathodeSensorSurface
    RSOilSurface							RSOilSurface
    ESRAirSurfaceTop					      <
    ESRAirSurfaceBot					      <
    AdCableTraySurface						AdCableTraySurface
    SSTWaterSurfaceNear2					      <
    PmtMtTopRingSurface						PmtMtTopRingSurface
    PmtMtBaseRingSurface						PmtMtBaseRingSurface
    PmtMtRib1Surface						PmtMtRib1Surface
    PmtMtRib2Surface						PmtMtRib2Surface
    PmtMtRib3Surface						PmtMtRib3Surface
    LegInIWSTubSurface						LegInIWSTubSurface
    TablePanelSurface						TablePanelSurface
    SupportRib1Surface						SupportRib1Surface
    SupportRib5Surface						SupportRib5Surface
    SlopeRib1Surface						SlopeRib1Surface
    SlopeRib5Surface						SlopeRib5Surface
    ADVertiCableTraySurface						ADVertiCableTraySurface
    ShortParCableTraySurface					ShortParCableTraySurface
    NearInnInPiperSurface						NearInnInPiperSurface



Yep the old one is following the sorted order from opticksdata, the direct isnt::

    In [1]: import json
    In [2]: o = json.load(file("export/DayaBay/GSurfaceLib/order.json"))
    In [3]: print "\n".join(["%3s : %s " % ( kv[1], kv[0]) for kv in sorted(o.items(), key=lambda kv:int(kv[1]))])
      1 : NearPoolCoverSurface 
      2 : NearDeadLinerSurface 
      3 : NearOWSLinerSurface 
      4 : NearIWSCurtainSurface 
      5 : SSTWaterSurfaceNear1 
      6 : SSTOilSurface 
      7 : lvPmtHemiCathodeSensorSurface 
      8 : lvHeadonPmtCathodeSensorSurface 
      9 : RSOilSurface 
     10 : ESRAirSurfaceTop 
     11 : ESRAirSurfaceBot 
     12 : AdCableTraySurface 
     13 : SSTWaterSurfaceNear2 
     14 : PmtMtTopRingSurface 
     15 : PmtMtBaseRingSurface 
     16 : PmtMtRib1Surface 
     17 : PmtMtRib2Surface 
     18 : PmtMtRib3Surface 
     19 : LegInIWSTubSurface 
     20 : TablePanelSurface 
     21 : SupportRib1Surface 
     22 : SupportRib5Surface 
     23 : SlopeRib1Surface 
     24 : SlopeRib5Surface 
     25 : ADVertiCableTraySurface 
     26 : ShortParCableTraySurface 
     27 : NearInnInPiperSurface 
     28 : NearInnOutPiperSurface 
     29 : LegInOWSTubSurface 
     30 : UnistrutRib6Surface 
     31 : UnistrutRib7Surface 
     32 : UnistrutRib3Surface 
     33 : UnistrutRib5Surface 
     34 : UnistrutRib4Surface 
     35 : UnistrutRib1Surface 
     36 : UnistrutRib2Surface 
     37 : UnistrutRib8Surface 
     38 : UnistrutRib9Surface 
     39 : TopShortCableTraySurface 
     40 : TopCornerCableTraySurface 
     41 : VertiCableTraySurface 
     42 : NearOutInPiperSurface 
     43 : NearOutOutPiperSurface 
     44 : LegInDeadTubSurface 







