Scintillation Photons
======================

::

   ggv-;ggv-g4gun
         # geant4 particle gun simulation within default DYB geometry, loaded from GDML
         # using 

OpNovicePhysicsList not yielding scintillation photons. Why?

::

    ggv-;ggv-g4gun --dbg

    (lldb) b "G4Scintillation::PostStepDoIt(G4Track const&, G4Step const&)" 
    Breakpoint 1: where = libG4processes.dylib`G4Scintillation::PostStepDoIt(G4Track const&, G4Step const&) + 39 at G4Scintillation.cc:194, address = 0x00000001035b7c87
    (lldb) b "G4Scintillation::AtRestDoIt(G4Track const&, G4Step const&)" 
    Breakpoint 2: where = libG4processes.dylib`G4Scintillation::AtRestDoIt(G4Track const&, G4Step const&) + 24 at G4Scintillation.cc:179, address = 0x00000001035b7c48

    (lldb) p TotalEnergyDeposit
    (G4double) $2 = 0.25795194278917472

    (lldb) p *aMaterialPropertiesTable    # only standard 4 + REEMISSIONPROB, no FASTCOMPONENT SLOWCOMPONENT 



* the ancient GDML export lacks material props, so added them in from geocache with CPropLib
* but scintillator props are handled differently, the reemiision buffer is 
  calculated and only that gets persisted in GScintillatorLib

* actually GMaterialLib only persists the material buffer also, but the import 
  reconstructs the individual GMaterial... but they are the standard props
  not the raw ones

* need to persist scintillator raw materials in geocache GScintillatorLib
  


Tracing thru Opticks
----------------------

::

    simon:ggeo blyth$ grep FASTCOMP *.*
    GGeo.cc:    findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB"); 
    GScintillatorLib.cc:"fast_component:FASTCOMPONENT," 

In GGeo (pre-cache) scintillator raw GMaterials are selected::

    948 std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, const char* delim)

And used to create the GScintillatorLib which is persisted in geocache::

     806 void GGeo::prepareScintillatorLib()
     807 {
     808     LOG(info) << "GGeo::prepareScintillatorLib " ;
     809 
     810     findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB");
     811 
     812     unsigned int nscint = getNumScintillatorMaterials() ;
     813 
     814     if(nscint == 0)
     815     {
     816         LOG(warning) << "GGeo::prepareScintillatorLib found no scintillator materials  " ;
     817     }
     818     else
     819     {
     820         GPropertyMap<float>* scint = dynamic_cast<GPropertyMap<float>*>(getScintillatorMaterial(0));
     821 
     822         GScintillatorLib* sclib = getScintillatorLib() ;
     823 
     824         sclib->add(scint);
     825 
     826         sclib->close();
     827     }
     828 }

But just the GPU ready buffer is persisted, not the scintillator materials.







