#pragma once

#include "plog/Severity.h"
#include "G4OK_API_EXPORT.hh"

class G4Run ; 
class G4Event ; 
class G4Track ; 
class G4Step ; 

struct G4OK_API G4OpticksRecorder 
{
    static const plog::Severity LEVEL ; 

    G4OpticksRecorder(); 
    virtual ~G4OpticksRecorder(); 

    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);

    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);

    virtual void PreUserTrackingAction(const G4Track*);
    virtual void PostUserTrackingAction(const G4Track*);

    virtual void UserSteppingAction(const G4Step*);

};



