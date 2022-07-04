#pragma once
/**
Temporarily : G4CXOpticks, Aiming to replace G4Opticks  
=========================================================

* KEEP THIS MINIMAL : PROVIDING TOP LEVEL INTERFACE AND COORDINARION 
* EVERYTHING THAT CAN BE IMPLEMENTED AT LOWER LEVELS SHOULD BE IMPLEMENTED AT LOWER LEVELS 

HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/

class G4VPhysicalVolume ;  

class GGeo ; 
struct CSGFoundry ; 
struct CSGOptiX ; 
struct QSim ; 

#include "plog/Severity.h"
#include "G4CX_API_EXPORT.hh"

struct G4CX_API G4CXOpticks
{
    static const plog::Severity LEVEL ;
    static std::string Desc();

    const G4VPhysicalVolume* wd ; 
    const GGeo*             gg ;
    CSGFoundry* fd ; 
    CSGOptiX*   cx ; 
    QSim*       qs ; 

 
    G4CXOpticks(); 
    std::string desc() const ; 

    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* wd); 
    void setGeometry(const GGeo* gg); 
    void setGeometry(CSGFoundry* fd); 

    void render(); 
    void simulate(); 
    void simtrace(); 
};








