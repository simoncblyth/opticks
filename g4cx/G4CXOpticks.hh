#pragma once
/**
Temporarily : G4CXOpticks, Aiming to replace G4Opticks  
=========================================================

* KEEP THIS MINIMAL : PROVIDING TOP LEVEL INTERFACE AND COORDINARION 
* EVERYTHING THAT CAN BE IMPLEMENTED AT LOWER LEVELS SHOULD BE IMPLEMENTED AT LOWER LEVELS 

  * IN THIS REGARD THE THREE PRIMARY METHODS simulate/simtrace/render ALL NEED OVERHAUL  
    TO INSTEAD RELAY ON CSGOptiX LEVEL 

HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/

struct U4Tree ; 
struct U4SensorIdentifier ; 
class G4VPhysicalVolume ;  

class  GGeo ; 
struct CSGFoundry ; 
struct CSGOptiX ; 
struct SSim ; 
struct QSim ; 

#include "schrono.h"
#include "plog/Severity.h"
#include "G4CX_API_EXPORT.hh"

struct G4CX_API G4CXOpticks
{
    static const plog::Severity LEVEL ;
    static const U4SensorIdentifier* SensorIdentifier ; 
    static void SetSensorIdentifier( const U4SensorIdentifier* sid ); 

    static G4CXOpticks* INSTANCE ;
    static G4CXOpticks* Get(); 
    static G4CXOpticks* SetGeometry() ; 
    static G4CXOpticks* SetGeometry(const G4VPhysicalVolume* world) ; 

    static constexpr const char* SaveGeometry_KEY = "G4CXOpticks__SaveGeometry_DIR" ; 
    static void SaveGeometry(); 
    static void Finalize(); 

    static bool NoGPU ; 
    static void SetNoGPU(bool no_gpu=true) ; // exercise everything other than CSGOptiX
    static bool IsNoGPU() ; 


    SSim*                    sim ; 
    const U4Tree*            tr ;
    const G4VPhysicalVolume* wd ; 
    GGeo*                    gg ;
    CSGFoundry*              fd ; 
    CSGOptiX*                cx ; 
    QSim*                    qs ; 
    schrono::TP              t0 ; 

    // WIP: move these into QSim
    //unsigned event_total ; 
    //unsigned genstep_total ; 
    //unsigned photon_total ; 
    //unsigned hit_total ; 

private: 
    G4CXOpticks(); 
    void init(); 
public: 
    virtual ~G4CXOpticks(); 

    static std::string Desc();
    std::string desc() const ; 

private: 
    void setGeometry(); 
    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* world);  
    void setGeometry(GGeo* gg); 
    static const char* setGeometry_saveGeometry ; 
    void setGeometry(CSGFoundry* fd); 
    void setGeometry_(CSGFoundry* fd); 
    void setupFrame(); 
    static const bool simulate_saveEvent ; // G4CXOpticks__simulate_saveEvent
public: 
    std::string descSimulate() const ; 
    void simulate(int eventID); 
    void simtrace(int eventID); 
    void render(); 

    void saveEvent() const ; // TODO: eliminate, handle it at SEvt level via SEventConfig control 

    static const bool saveGeometry_saveGGeo ; // G4CXOpticks__saveGeometry_saveGGeo 
    void saveGeometry() const ;
    void saveGeometry(const char* dir) const ; 

};

