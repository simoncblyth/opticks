full_geom_missing_ellipsoid_transform_again
=============================================



Issue
---------

Upper hemisphere in full GDML running seems to miss the ellipsoid transform. 

* failed to reproduce with simpler geometry hama_body_log so far 
* also failed to reproduce with GDMLSub running with HamamatsuR12860sMask_virtual, 
  placing a single "hatbox" + mask + PMT  


* TODO: try adding some instance transforms to duplicate the grabbed 
  GDMLSub and see if the transform troubles manifest then 




DONE : Using SPlace.h transforms with G4PVPlacement
-----------------------------------------------------

Added::

    U4VolumeMaker::WrapInstance
    U4VolumeMaker::MakeTransforms
    U4VolumeMaker::WrapAround
     

::

    341 /**
    342 U4VolumeMaker::WrapInstance
    343 -----------------------------
    344 
    345 The *item_lv* is repeated many times using transforms from U4VolumeMaker::MakeTransforms
    346 which makes use of SPlace::AroundSphere SPlace::AroundCylinder
    347 
    348 All those repeats have a "Water" box mother volume which is contained within "Rock". 
    349 
    350 **/
    351 
    352 G4VPhysicalVolume* U4VolumeMaker::WrapInstance( G4LogicalVolume* item_lv, const char* prefix )
    353 {
    354     const NP* trs = MakeTransforms(prefix) ;
    355     
    356     LOG(LEVEL)
    357         << " prefix " << prefix
    358         << " trs " << ( trs ? trs->sstr() : "-" )
    359         ;
    360         
    361     G4LogicalVolume*  rock_lv  = Box_(20000., "Rock" );
    362     G4LogicalVolume*  water_lv = Box_(19000., "Water" );
    363     
    364     WrapAround(prefix, trs, item_lv, water_lv );
    365     
    366     G4VPhysicalVolume* water_pv = Place(water_lv,  rock_lv);  assert( water_pv );
    367     G4VPhysicalVolume* rock_pv  = Place(rock_lv,  nullptr );  
    368     
    369     return rock_pv ;
    370 }   




Several different ctors, might be easier to use the one which splits
rotation from translation as the G4Transform3D arg.::

     71 G4PVPlacement::G4PVPlacement( const G4Transform3D &Transform3D,
     72                               const G4String& pName,
     73                                     G4LogicalVolume *pLogical,
     74                                     G4VPhysicalVolume *pMother,
     75                                     G4bool pMany,
     76                                     G4int pCopyNo,
     77                                     G4bool pSurfChk )
     78   : G4VPhysicalVolume(NewPtrRotMatrix(Transform3D.getRotation().inverse()),
     79                       Transform3D.getTranslation(),pName,pLogical,pMother),
     80     fmany(pMany), fcopyNo(pCopyNo)
     81 {
     



DONE : SPlace.h creates instance transforms AroundCylinder or AroundSphere
----------------------------------------------------------------------------


Using GDMLSub machinery
--------------------------

geom::

     42 if [ "$GEOM" == "J000" ]; then
     43     export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml
     44     export J000_GDMLSub=HamamatsuR12860sMask_virtual0x:0:1000

oip::

     32 if [ "$GEOM" == "J000" ]; then
     33    if [ -z "$J000_GDMLSub" ]; then  ## need to switch off the _FRAME when using GDMLSub
     34        OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000
     35    fi
     36 fi



Added GDMLSub machinery for selecting volume from full GDML U4Volume::FindPVSub
---------------------------------------------------------------------------------

Develop machinery for loading full GDML and then selecting a volume::

    125 G4VPhysicalVolume* U4VolumeMaker::PVG_(const char* name)
    126 {   
    127     const char* gdmlpath = SOpticksResource::GDMLPath(name) ;
    128     const char* gdmlsub = SOpticksResource::GDMLSub(name);
    129     bool exists = gdmlpath && SPath::Exists(gdmlpath) ;
    130     
    131     G4VPhysicalVolume* loaded = exists ? U4GDML::Read(gdmlpath) : nullptr ;
    132     
    133     if(loaded && SSys::getenvbool(PVG_WriteNames))
    134         U4Volume::WriteNames( loaded, SPath::Resolve("$TMP", PVG_WriteNames, DIRPATH));
    135     
    136     G4VPhysicalVolume* pv = loaded ;
    137     
    138     if( loaded && gdmlsub )
    139     {   
    140         G4VPhysicalVolume* pv_sub = U4Volume::FindPVSub( loaded, gdmlsub ) ;
    141         G4LogicalVolume* lv_sub = pv_sub->GetLogicalVolume(); 
    142         pv = WrapRockWater( lv_sub ) ;           // HMM: assuming the gdmlsub is in Water ?
    143         LOG(LEVEL) << " WrapRockWater lv_sub " << ( lv_sub ? lv_sub->GetName() : "-" );
    144     }
    145     


As preliminary to that added U4Volume::WriteNames. Check PLS.txt from two runs, 
shows are getting same pointers in the names because coming from the GDML::

     212895 solidXJfixture0x595eb40
     212896 pLPMT_Hamamatsu_R128600x5f67fb0
     212897 HamamatsuR12860lMaskVirtual0x5f51160
     212898 HamamatsuR12860sMask_virtual0x5f50520
     212899 HamamatsuR12860pMask0x5f51db0
     212900 HamamatsuR12860lMask0x5f51c50
     212901 HamamatsuR12860sMask0x5f51a40
     212902 HamamatsuR12860pMaskTail0x5f53210


     212895 solidXJfixture0x595eb40
     212896 pLPMT_Hamamatsu_R128600x5f67fb0
     212897 HamamatsuR12860lMaskVirtual0x5f51160
     212898 HamamatsuR12860sMask_virtual0x5f50520
     212899 HamamatsuR12860pMask0x5f51db0
     212900 HamamatsuR12860lMask0x5f51c50
     212901 HamamatsuR12860sMask0x5f51a40
     212902 HamamatsuR12860pMaskTail0x5f53210
     212903 HamamatsuR12860lMaskTail0x5f530c0
     212904 HamamatsuR12860Tail0x5f52eb0




