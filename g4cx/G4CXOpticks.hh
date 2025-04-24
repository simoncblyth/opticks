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
struct NPFold ; 
struct NP ; 
struct U4SensorIdentifier ; 
class G4VSensitiveDetector ; 
class G4VPhysicalVolume ;  
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
    static U4SensorIdentifier* SensorIdentifier ; 
    static void SetSensorIdentifier( U4SensorIdentifier* sid ); 

    static G4CXOpticks* INSTANCE ;
    static G4CXOpticks* Get(); 
    static G4CXOpticks* SetGeometry() ; 
    static G4CXOpticks* SetGeometryFromGDML() ; 
    static G4CXOpticks* SetGeometry(const G4VPhysicalVolume* world) ; 
    static G4CXOpticks* SetGeometry_JUNO(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd, NPFold* jpmt, const NP* jlut ) ; 


    static constexpr const char* SaveGeometry_KEY = "G4CXOpticks__SaveGeometry_DIR" ; 
    static void SaveGeometry(); 
    static void Finalize(); 

    static bool NoGPU ; 
    static void SetNoGPU(bool no_gpu=true) ; // exercise everything other than CSGOptiX
    static bool IsNoGPU() ; 


    SSim*                    sim ; 
    const U4Tree*            tr ;
    const G4VPhysicalVolume* wd ; 
    CSGFoundry*              fd ; 
    CSGOptiX*                cx ; 
    QSim*                    qs ; 
    schrono::TP              t0 ; 


private: 
    G4CXOpticks(); 
    void init(); 
public: 
    virtual ~G4CXOpticks(); 

    static std::string Desc();
    std::string desc() const ; 

private: 
    void setGeometry(); 
    void setGeometryFromGDML(); 
    void setGeometry(const char* gdmlpath);
    void setGeometry(const G4VPhysicalVolume* world);  
    static const char* setGeometry_saveGeometry ; 
    void setGeometry(CSGFoundry* fd); 
    void setGeometry_(CSGFoundry* fd); 
public: 
    std::string descSimulate() const ; 

    void simulate( int eventID, bool reset ); 
    void reset(    int eventID ); 

    void simtrace(int eventID); 
    void render(); 

    void saveGeometry() const ;
    void saveGeometry(const char* dir) const ; 


    void SensitiveDetector_Initialize(int eventID);
    void SensitiveDetector_EndOfEvent(int eventID); 

};

