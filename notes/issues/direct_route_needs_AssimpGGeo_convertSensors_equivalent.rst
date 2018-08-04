direct_route_needs_AssimpGGeo_convertSensors_equivalent
=========================================================

Strategy
---------

In coming up with an equivalent for the old way, dont let the old way constrain
you : adopt the simplest approach for the direct route.

Having the real G4 model to probe, not just the pale reflection of GDML/G4DAE
gives more possibilities.

BUT : am testing with a geometry loaded from GDML, and have to keep 
feeding in the truth from G4DAE to fix missing things like material
props and surfaces.  Which makes sticking very close to the old way 
more appealing.


Issue
------

Want to test the direct route from a Geant4 world, but 
booting from GDML start with:

1. no material properties : used CGDMLDetector::addMPT

2. no surfaces : used CDetector::attachSurfaces to 
   convert the Opticks surfaces into G4 ones with the fake
   SensorSurfaces skipped

3. no sensors 
  
Have fixed up using CFG4 machinery 
but trying to fix sensors runs into problem of 
Opticks/Geant4 model mismatch : Opticks puts 
detection efficiency property onto artificially 
added SensorSurface which is ascribed to the cathode LV. 

Need to mock up the original situation with EFFICIENCY 
property on the Cathode Bialkali material.

Problem is complicated:

1. have to mock up the Geant4 model from GDML + Opticks 
   model fixups for deficiencies (this is why CFG4 is very useful
   as it has lots of conversions from Opticks to G4)

2. apply X4 direct machinery to convert from G4 model back to Opticks 

   X4PhysicalVolume::convertSensors
   GGeoSensor::AddSensorSurfaces



CPropLib::makeMaterialPropertiesTable maybe fixed this already
-----------------------------------------------------------------

::

    201 G4MaterialPropertiesTable* CPropLib::makeMaterialPropertiesTable(const GMaterial* ggmat)
    202 {
    203     const char* name = ggmat->getShortName();
    204     GMaterial* _ggmat = const_cast<GMaterial*>(ggmat) ; // wont change it, i promise 
    205 
    206     LOG(error) << " name " << name ;
    207 
    208 
    209     G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    210     addProperties(mpt, _ggmat, "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB,GROUPVEL");
    211 
    212     if(strcmp(name, SENSOR_MATERIAL)==0)
    213     {
    214         GPropertyMap<float>* surf = m_sensor_surface ;
    215 
    216         if(!surf)
    217         {
    218             LOG(fatal) << "CPropLib::makeMaterialPropertiesTable"
    219                        << " material with SENSOR_MATERIAL name " << name
    220                        << " but no sensor_surface "
    221                        ;
    222             LOG(fatal) << "m_sensor_surface is obtained from slib at CPropLib::init "
    223                        << " when Bialkai material is in the mlib "
    224                        << " it is required for a sensor surface (with EFFICIENCY/detect) property "
    225                        << " to be in the slib "
    226                        ;
    227         }
    228         assert(surf);
    229         addProperties(mpt, surf, "EFFICIENCY");
    230 
    231         // REFLECTIVITY ?
    232     }





How about being radically simple : just require substring "cathode" or "Cathode" in the LV name 
------------------------------------------------------------------------------------------------

* this avoids sensor list and getting people to make one, and update it as geometry changes 

  * PMT identifiers can come later, i can just give them an adhoc index  


CGDMLDetector::addMPT
----------------------

Have observed, Bialkali looses its EFFICIENCY, once thru Opticks standardization, the
EFFICIENCY gets planted onto the fake SensorSurfaces::

    2018-08-03 15:32:55.668 ERROR [7953232] [X4Material::init@99] name Bialkali
    2018-08-03 15:32:55.668 ERROR [7953232] [X4MaterialPropertiesTable::AddProperties@41] ABSLENGTH
    2018-08-03 15:32:55.669 ERROR [7953232] [X4MaterialPropertiesTable::AddProperties@41] GROUPVEL
    2018-08-03 15:32:55.669 ERROR [7953232] [X4MaterialPropertiesTable::AddProperties@41] RAYLEIGH
    2018-08-03 15:32:55.669 ERROR [7953232] [X4MaterialPropertiesTable::AddProperties@41] REEMISSIONPROB
    2018-08-03 15:32:55.669 ERROR [7953232] [X4MaterialPropertiesTable::AddProperties@41] RINDEX


Fixed inconsistency by moving surface collection from GGeo entirely into GSurfaceLib::

    2018-08-03 18:40:13.570 INFO  [8076604] [X4PhysicalVolume::convertSurfaces@252] convertSurfaces num_lbs 8 num_sks 34
    2018-08-03 18:40:13.603 FATAL [8076604] [GGeoSensor::AddSensorSurfaces@31]  require a cathode material to AddSensorSurfaces 
    2018-08-03 18:40:13.603 ERROR [8076604] [X4PhysicalVolume::convertSensors@167]  m_lvsdname PmtHemiCathode,HeadonPmtCathode num_clv 2 num_bds 0 num_sks0 0 num_sks1 0
    2018-08-03 18:40:13.603 INFO  [8076604] [GPropertyLib::close@418] GPropertyLib::close type GSurfaceLib buf 46,2,39,4

    2018-08-03 19:39:59.246 INFO  [8108243] [X4PhysicalVolume::convertSurfaces@252] convertSurfaces num_lbs 8 num_sks 34
    2018-08-03 19:39:59.279 FATAL [8108243] [GGeoSensor::AddSensorSurfaces@31]  require a cathode material to AddSensorSurfaces 
    2018-08-03 19:39:59.279 ERROR [8108243] [X4PhysicalVolume::convertSensors@167]  m_lvsdname PmtHemiCathode,HeadonPmtCathode num_clv 2 num_bds 8 num_sks0 34 num_sks1 34
    2018-08-03 19:39:59.279 INFO  [8108243] [GPropertyLib::close@418] GPropertyLib::close type GSurfaceLib buf 46,2,39,4




Finding Sensitive Volumes  : old way 
---------------------------------------

* combined a material with an EFFICIENCY property with a sensor list of volumes 

  * dumping props are getting loadsa EFFICIENCY (some default addition ?) 
  * dont like the need to have a sensor list of volume indices , although it 
    does have advantage of input of sensor identifiers 


Somehow all surfaces have EFFICIENCY + REFLECTIVITY 
-------------------------------------------------------

* so EFFICIENCY prop doent help to find cathode LV

::

    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] REFLECTIVITY
    2018-08-02 20:21:56.640 INFO  [7626612] [X4LogicalBorderSurfaceTable::init@38]  src ESRAirSurfaceBot
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] EFFICIENCY
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] REFLECTIVITY
    2018-08-02 20:21:56.640 INFO  [7626612] [X4LogicalBorderSurfaceTable::init@38]  src SSTWaterSurfaceNear2
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] EFFICIENCY
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] REFLECTIVITY
    2018-08-02 20:21:56.640 ERROR [7626612] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces num_src 36
    2018-08-02 20:21:56.640 INFO  [7626612] [X4LogicalSkinSurfaceTable::init@38]  src NearPoolCoverSurface
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] EFFICIENCY
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] REFLECTIVITY
    2018-08-02 20:21:56.640 INFO  [7626612] [X4LogicalSkinSurfaceTable::init@38]  src lvPmtHemiCathodeSensorSurface
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] EFFICIENCY
    2018-08-02 20:21:56.640 ERROR [7626612] [X4MaterialPropertiesTable::AddProperties@41] REFLECTIVITY
    2018-08-02 20:21:56.640 INFO  [7626612] [X4LogicalSkinSurfaceTable::init@38]  src lvHeadonPmtCathodeSensorSurface




G4 SD Review
-------------

Maybe can just check all logvols::

   G4VSensitiveDetector* originalSD = logVol->GetSensitiveDetector(); 
   ## but does that survive GDML ? 

::

    050 class G4VUserDetectorConstruction
     51 { 
     80   protected:
     81     void SetSensitiveDetector(const G4String& logVolName,
     82                 G4VSensitiveDetector* aSD,G4bool multi=false);
     83     void SetSensitiveDetector(G4LogicalVolume* logVol,
     84                 G4VSensitiveDetector* aSD);
     85 };


    239 void G4VUserDetectorConstruction::SetSensitiveDetector
    240 (G4LogicalVolume* logVol, G4VSensitiveDetector* aSD)
    241 {
    242   assert(logVol!=nullptr&&aSD!=nullptr);
    243 
    244   G4SDManager::GetSDMpointer()->AddNewDetector(aSD);
    245 
    246   //New Logic: allow for "multiple" SDs being attached to a single LV.
    247   //To do that we use a special proxy SD called G4MultiSensitiveDetector
    248 
    249   //Get existing SD if already set and check if it is of the special type
    250   G4VSensitiveDetector* originalSD = logVol->GetSensitiveDetector();
    251   if ( originalSD == nullptr ) {
    252       logVol->SetSensitiveDetector(aSD);
    253   } else {
    254       G4MultiSensitiveDetector* msd = dynamic_cast<G4MultiSensitiveDetector*>(originalSD);
    255       if ( msd != nullptr ) {
    256           msd->AddSD(aSD);
    257       } else {
    258           const G4String msdname = "/MultiSD_"+logVol->GetName();
    259           msd = new G4MultiSensitiveDetector(msdname);
    260           //We need to register the proxy to have correct handling of IDs
    261           G4SDManager::GetSDMpointer()->AddNewDetector(msd);
    262       msd->AddSD(originalSD);
    263           msd->AddSD(aSD);
    264           logVol->SetSensitiveDetector(msd);
    265       }
    266   }
    267 }



Adding SensorSurfaces
-----------------------

::

    epsilon:geant4_10_02_p01 blyth$ g4-cc EFFICIENCY 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:              aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/global/HEPNumerics/src/G4ConvergenceTester.cc:   out << std::setw(20) << "EFFICIENCY = " << std::setw(13)  << efficiency << G4endl;
    epsilon:geant4_10_02_p01 blyth$ 
    epsilon:geant4_10_02_p01 blyth$ 
    epsilon:geant4_10_02_p01 blyth$ g4-hh EFFICIENCY 
    epsilon:geant4_10_02_p01 blyth$ 

