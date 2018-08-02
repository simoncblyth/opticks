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


How about being radically simple : just require substring "cathode" or "Cathode" in the LV name 
------------------------------------------------------------------------------------------------

* this avoids sensor list and getting people to make one, and update it as geometry changes 

  * PMT identifiers can come later, i can just give them an adhoc index  



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

