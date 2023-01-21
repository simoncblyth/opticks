#pragma once
/**
C4Solid.hh : G4VSolid -> CSGPrim (which references sequence of CSGNode) 
=========================================================================

Priors to grab from:

U4SolidTree 
    G4 tree mechanics

X4Solid 
    G4VSolid param extraction 

CSG_GGeo_Convert 
    mechanics of CSGPrim/CSGNode creation : eg direct into CSGFoundry ? 

    CSG_GGeo_Convert::convertPrim would suggest better to start more centralized
    (from the top) around CSGFoundry and farm off small parts where appropriate


**/

class G4VSolid ; 
struct CSGPrim ; 

#include "C4_API_EXPORT.hh"

struct C4_API C4Solid
{
    static CSGPrim* Convert(const G4VSolid* solid) ; 
}; 




