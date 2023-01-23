#pragma once
/**
C4.hh : Geometry Translation without the GGeo intermediate model
====================================================================

* C4 is short for "CSG_U4"

* U4/U4Tree has done lots of the heavy lifting already, continue 
  doing as much of the translation impl in U4 as possible 
  with only the final stage that needs CSG done up here 

* Also continue the pattern from U4Tree of keeping persist types
  at lower sysrap dependency level eg SSim/stree/NP/NPFold


::

    .       C4   
    Geant4 --->  CSG/CSGFoundry 


            X4      CSG_GGeo
    Geant4 --->  GGeo --->  CSG/CSGFoundry 



* All functionality exists already, just have to reorganize
  what already exists in a simpler and much less code manner. 

* This is an extreme code reduction exercise removing dependency on the packages::

   BRAP
   NPY
   OKC
   ExtG4
   GGEO
   CSG_GGeo


**/
#include <string>
#include "C4_API_EXPORT.hh"

class G4VPhysicalVolume ; 
struct U4SensorIdentifier ; 
struct SSim ; 
struct stree ; 
struct U4Tree ; 
struct CSGFoundry ; 

struct C4_API C4
{
    static const U4SensorIdentifier* SensorIdentifier ;
    static void SetSensorIdentifier( const U4SensorIdentifier* sid );

    const G4VPhysicalVolume* top ; 
    SSim*       sim ; 
    stree*      st ; 
    U4Tree*     tr ; 
    CSGFoundry* fd ; 

    static CSGFoundry* Translate(const G4VPhysicalVolume* const top ); 

    C4(const G4VPhysicalVolume* const top); 
    void init(); 

    std::string desc() const ; 
}; 


