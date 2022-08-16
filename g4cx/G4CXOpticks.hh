#pragma once
/**
Temporarily : G4CXOpticks, Aiming to replace G4Opticks  
=========================================================

* KEEP THIS MINIMAL : PROVIDING TOP LEVEL INTERFACE AND COORDINARION 
* EVERYTHING THAT CAN BE IMPLEMENTED AT LOWER LEVELS SHOULD BE IMPLEMENTED AT LOWER LEVELS 

HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/

struct U4Tree ; 
struct U4SensorIdentifier ; 
class G4VPhysicalVolume ;  

class  GGeo ; 
struct CSGFoundry ; 
struct CSGOptiX ; 
struct QSim ; 

#include "plog/Severity.h"
#include "G4CX_API_EXPORT.hh"

struct G4CX_API G4CXOpticks
{
    static const plog::Severity LEVEL ;
    static const U4SensorIdentifier* SensorIdentifier ; 
    static void SetSensorIdentifier( const U4SensorIdentifier* sid ); 

    static G4CXOpticks* INSTANCE ; 
    static G4CXOpticks* Get(); 
    static constexpr const char* RELDIR = "G4CXOpticks" ; 
    static std::string FormPath(const char* base, const char* rel );
    static void SetGeometry(const G4VPhysicalVolume* world) ; 
    static void Finalize(); 

    const U4Tree*   tr ;
    const G4VPhysicalVolume* wd ; 

    GGeo*       gg ;
    CSGFoundry* fd ; 
    CSGOptiX*   cx ; 
    QSim*       qs ; 
 
    G4CXOpticks(); 

    static std::string Desc();
    std::string desc() const ; 

    void setGeometry(); 
    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* world);  
    void setGeometry(GGeo* gg); 
    void setGeometry(CSGFoundry* fd); 

    void render(); 

    static const bool simulate_saveEvent ; 
    void simulate(); 
    void simtrace(); 

    void saveEvent() const ; 

    void saveGeometry() const ;
    void saveGeometry(const char* base, const char* rel=nullptr) const; 
    void saveGeometry_(const char* dir) const ; 


};

