#pragma once
/**
Temporarily : G4CXOpticks, Aiming to replace G4Opticks  
=========================================================


HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/

class GGeo ; 
struct CSGFoundry ; 
struct CSGOptiX ; 
class G4VPhysicalVolume ;  

#include "plog/Severity.h"
#include "G4CX_API_EXPORT.hh"

struct G4CX_API G4CXOpticks
{
    static const plog::Severity LEVEL ;

    const G4VPhysicalVolume* wd ; 
    const GGeo*             gg ;
    CSGFoundry* fd ; 
    CSGOptiX*   cx ; 

    G4CXOpticks(); 

    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* wd); 
    void setGeometry(const GGeo* gg); 
    void setGeometry(CSGFoundry* fd); 

    void render_snap(); 

};








