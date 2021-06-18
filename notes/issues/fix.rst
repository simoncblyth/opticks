fix : Brief Summaries of Recent Fixes/Additions 
==================================================

* G4OpticksRecorder/CRecorder machinery working with dynamic(genstep-by-genstep) running
  by dynamically expanding output arrays at each BeginOfGenstep  

* CPhotonInfo overhaul for cross reemission generation recording 

* suppress PMT Pyrex G4 0.001 mm "microsteps"

* handle RINDEX-NoRINDEX "ImplicitSurface" transitions like Water->Tyvek by adding corresponding Opticks perfect absorber surfaces

* handle G4 "Water" RAYLEIGH special casing, by grabbing the calulated RAYLEIGH and adding to material, see X4MaterialWater









