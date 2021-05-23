#pragma once

#include "plog/Severity.h"
#include "G4OK_API_EXPORT.hh"

class G4Run ; 
class G4Event ; 
class G4Track ; 
class G4Step ; 

class GGeo ; 
class Opticks ; 
class CMaterialBridge ; 

struct CManager ; 



/**
G4OpticksRecorder 
====================

Objective : write Geant4 propagations into OpticksEvent format arrays, 
reusing classes from CFG4/CRecorder machinery where possible.  

**/

struct G4OK_API G4OpticksRecorder 
{
    static const plog::Severity LEVEL ; 
    static G4OpticksRecorder* Get() ; 

    const GGeo*      m_ggeo ; 
    Opticks*         m_ok ; 
    CMaterialBridge* m_material_bridge ; 
    CManager*        m_manager ; 


    G4OpticksRecorder(); 
    virtual ~G4OpticksRecorder(); 
    void setGeometry(const GGeo* ggeo); 

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    virtual void PreUserTrackingAction(const G4Track*);
    virtual void PostUserTrackingAction(const G4Track*);

    virtual void UserSteppingAction(const G4Step*);

    static G4OpticksRecorder*  fInstance;
};



