#pragma once

#include "G4OK_API_EXPORT.hh"

class Opticks;
class GGeo ; 

/**
G4OpticksGeometry
==================

Exploration of possible direct geometry conversion from live G4 
in memory to Opticks GGeo/GScene etc.. With no intermediary G4DAE/GDML export 
file.  Although GLTF could be written from the GScene.


**/

class G4OK_API G4OpticksGeometry 
{
  public:
    G4OpticksGeometry();
    static int load(GGeo* ggeo) ; 


};
 
