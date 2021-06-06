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
G4OpticksRecorder : Optionally Performs Geant4 Simulation with propagations saved into "g4evt" OpticksEvent
==================================================================================================================

This is used from the **OPTIONAL** JUNO code G4OpticksAnaMgr

The recorder receives Geant4 objects : G4Run, G4Event, G4Track, G4Step 
and writes full step-by-step photon histories of all photons 
into OpticksEvent format arrays, using Opticks CFG4/CRecorder/CWriter machinery. 

Note that the setGeometry and setGenstep, endGenstep methods
are invoked from G4Opticks via G4OpticksRecorder::Get() 
only when there is G4OpticksRecorder instance. 

**/

struct G4OK_API G4OpticksRecorder 
{
    static const plog::Severity LEVEL ; 
    static G4OpticksRecorder* Get() ; 


    const GGeo*      m_ggeo ; 
    Opticks*         m_ok ; 
    CMaterialBridge* m_material_bridge ; 
    CManager*        m_manager ; 

public:

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

private:
    void TerminateEvent();

};



