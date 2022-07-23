update_juno_opticks_integration_for_new_workflow
==================================================

* previous : :doc:`ellipsoid_transform_compare_two_geometries`

How to simplify integration ?
-----------------------------

* Do not return G4(CX)Opticks instance, so can just change impl not header 
* Keep it totally minimal : ie do everything on Opticks side and the 
  absolute minimum on the Detector Framework side


Old Integration : setup : done at tail of LSExpDetectorConstruction::Construct
---------------------------------------------------------------------------------

jcv LSExpDetectorConstruction::


     199 G4VPhysicalVolume* LSExpDetectorConstruction::Construct()
     200 {
     ...
     359   m_g4opticks = LSExpDetectorConstruction_Opticks::Setup( physiWorld, m_sd, m_opticksMode );
     360 
     361   G4cout
     362       << __FILE__ << ":" << __LINE__ << " completed construction of physiWorld "
     363       << " m_opticksMode " << m_opticksMode
     364       << G4endl
     365       ;
     366 
     367   return physiWorld;
     368 }


jcv LSExpDetectorConstruction_Opticks::

    001 #pragma once
      2 
      3 class G4Opticks ;
      4 class G4VPhysicalVolume ;
      5 class G4VSensitiveDetector ;
      6 
      7 struct LSExpDetectorConstruction_Opticks
      8 {
      9     static G4Opticks* Setup(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd_, int opticksMode );
     10 };

    084 G4Opticks* LSExpDetectorConstruction_Opticks::Setup(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd_, int opticksMode )  // static
     85 {
     86     if( opticksMode == 0 ) return nullptr ;
     87     LOG(info) << "[ WITH_G4OPTICKS opticksMode " << opticksMode  ;
     88 
     89     assert(world); 
     90 
     91     // 1. pass geometry to Opticks, translate it to GPU and return sensor placements  
     92 
     93     G4Opticks* g4ok = new G4Opticks ;
     94     
     95     bool outer_volume = true ;
     96     bool profile = true ;
     97 
     98     const char* geospecific_default =   "--way --pvname pAcrylic --boundary Water///Acrylic --waymask 3 --gdmlkludge" ;  // (1): gives radius 17820
     99     const char* embedded_commandline_extra = SSys::getenvvar("LSXDC_GEOSPECIFIC", geospecific_default ) ;   
    100     LOG(info) << " embedded_commandline_extra " << embedded_commandline_extra ;
    101 
    102     g4ok->setPlacementOuterVolume(outer_volume); 
    103     g4ok->setProfile(profile); 
    104     g4ok->setEmbeddedCommandLineExtra(embedded_commandline_extra);
    105     g4ok->setGeometry(world); 
    106 
    107     const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;       
    108     unsigned num_sensor = sensor_placements.size(); 
    109 
    110     // 2. use the placements to pass sensor data : efficiencies, categories, identifiers  
    111 
    112     const junoSD_PMT_v2* sd = dynamic_cast<const junoSD_PMT_v2*>(sd_) ;  
    113     assert(sd) ; 


    0596 void G4Opticks::setGeometry(const G4VPhysicalVolume* world)
     597 {
     598     LOG(LEVEL) << "[" ;
     599 
     600     LOG(LEVEL) << "( translateGeometry " ;
     601     GGeo* ggeo = translateGeometry( world ) ;
     602     LOG(LEVEL) << ") translateGeometry " ;
     603 
     604     if( m_standardize_geant4_materials )
     605     {
     606         standardizeGeant4MaterialProperties();
     607     }
     608 
     609     m_world = world ;
     610 
     611     setGeometry(ggeo);
     612 
     613     LOG(LEVEL) << "]" ;
     614 }

     940 GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
     941 {
     942     LOG(verbose) << "( key" ;
     943     const char* keyspec = X4PhysicalVolume::Key(top) ;
     944 
     945     bool parse_argv = false ;
     946     Opticks* ok = InitOpticks(keyspec, m_embedded_commandline_extra, parse_argv );
     947 
     948     // ok->setGPartsTransformOffset(true);  
     949     // HMM: CANNOT DO THIS PRIOR TO pre-7 
     950     // IDEA: COULD CREATE GParts TWICE WITH THE DIFFERENT SETTING AFTER pre-7 OGeo 
     951     // ACTUALLY: IT MAKES MORE SENSE TO SAVE IT ONLY IN CSG_GGeo : 
     952 
     953     const char* dbggdmlpath = ok->getDbgGDMLPath();
     954     if( dbggdmlpath != NULL )
     955     {
     956         LOG(info) << "( CGDML" ;
     957         CGDML::Export( dbggdmlpath, top );
     958         LOG(info) << ") CGDML" ;
     959     }

Old Integration : usage 
--------------------------

jcv junoSD_PMT_v2::

    1070 void junoSD_PMT_v2::EndOfEvent(G4HCofThisEvent* HCE)
    1071 {
    1072 
    1073 #ifdef WITH_G4OPTICKS
    1074     if(m_opticksMode > 0)
    1075     {
    1076         // Opticks GPU optical photon simulation and bulk hit population is done here 
    1077         m_jpmt_opticks->EndOfEvent(HCE);
    1078     }
    1079 #endif

jcv junoSD_PMT_v2_Opticks::
