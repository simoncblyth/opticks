#pragma once
/**
Temporarily : G4CXOpticks, Aiming to replace G4Opticks  
=========================================================

* KEEP THIS MINIMAL : PROVIDING TOP LEVEL INTERFACE AND COORDINARION 
* EVERYTHING THAT CAN BE IMPLEMENTED AT LOWER LEVELS SHOULD BE IMPLEMENTED AT LOWER LEVELS 

HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/

struct U4Sensor ; 
struct U4Tree ; 
class G4VPhysicalVolume ;  

class  GGeo ; 
struct CSGFoundry ; 
struct CSGOptiX ; 
struct QSim ; 
struct SEvt ; 

#include "plog/Severity.h"
#include "G4CX_API_EXPORT.hh"

struct G4CX_API G4CXOpticks
{
    static const plog::Severity LEVEL ;
    static G4CXOpticks* INSTANCE ; 
    static G4CXOpticks* Get(); 
    static constexpr const char* RELDIR = "G4CXOpticks" ; 
    static std::string FormPath(const char* base, const char* rel );
    static void SetGeometry(const G4VPhysicalVolume* world) ; 
    static void Finalize(); 

    const U4Sensor* sd ; 
    const U4Tree*   tr ;
    const G4VPhysicalVolume* wd ; 

    GGeo*       gg ;
    CSGFoundry* fd ; 
    CSGOptiX*   cx ; 
    QSim*       qs ; 
    SEvt*       se ; 
 
    G4CXOpticks(); 

    static std::string Desc();
    std::string desc() const ; 

    void setSensor(const U4Sensor* sd );

    void setGeometry(); 
    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* world);  
    // HMM: maybe add U4Sensor arg here, 
    // if that only makes sense at this level of geometry representation ?

    void setGeometry(GGeo* gg); 
    void setGeometry(CSGFoundry* fd); 

    void render(); 
    void simulate(); 
    void simtrace(); 

    void save() const ; 


    void saveGeometry() const ;
    void saveGeometry(const char* base, const char* rel=nullptr) const; 
    void saveGeometry_(const char* dir) const ; 


};

