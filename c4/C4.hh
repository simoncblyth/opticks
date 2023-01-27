#pragma once
/**
C4.hh : Geometry Translation without the GGeo intermediate model
====================================================================

* NOW LOOKING LIKE C4 PACKAGE WILL NOT BE NEEDED 

  * U4Tree will soon be able to populate stree.h with all the geometry info
  * then can populate CSGFoundry entirely from stree.h using a new "CSG/CSGFromTree" 
  * THUS LOOKS LIKE NO NEED FOR A PACKAGE DEPENDING ON BOTH Geant4 AND CSG 
  * AN INTERMEDIARY IS STILL NEEDED : stree.h/snd/snode EG FOR COINCIDENCE AVOIDANCE  

    * BUT IT CAN BE MINIMAL 
    * BENEFIT FROM PRE-KNOWLEDGE OF ENDPOINT CSG/CSGFoundry MODEL  


::

    .     U4/U4Tree            CSG
    Geant4  --> sysrap/stree.h --> CSG/CSGFoundry
                 (minimal)

    .       C4   
    Geant4 --->  CSG/CSGFoundry 


            X4      CSG_GGeo
    Geant4 --->  GGeo --->  CSG/CSGFoundry 
                (full)






* C4 is short for "CSG_U4"

* U4/U4Tree has done lots of the heavy lifting already, continue 
  doing as much of the translation impl in U4 as possible 
  with only the final stage that needs CSG done up here 

  * NO NEED : CAN DO IT WITHIN CSG AS ALL INFO GOES TO stree.h 

* Also continue the pattern from U4Tree of keeping persist types
  at lower sysrap dependency level eg SSim/stree/NP/NPFold



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


