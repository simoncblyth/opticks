fix : Brief Summaries of Recent Fixes/Additions 
==================================================

* G4OpticksRecorder/CRecorder machinery reworked to operate with dynamic(genstep-by-genstep) running by expanding 
  output arrays at each BeginOfGenstep  

* CPhotonInfo overhaul for cross reemission generation recording 

* suppress PMT Pyrex G4 0.001 mm "microsteps"

* handle RINDEX-NoRINDEX "ImplicitSurface" transitions like Water->Tyvek by adding corresponding Opticks perfect absorber surfaces

* handle Geant4 special casing of material with name "Water" that has RINDEX property but lacks RAYLEIGH
  by grabbing the calulated RAYLEIGH and adding to material (Geant4 only changes G4OpRayleigh process physics table) 
  see X4MaterialWater

* suppress degenerate Pyrex///Pyrex +0.001mm boundary in GPU geometry that caused material inconsistencies 









