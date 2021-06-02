#pragma once 

#include "CManager.hh"

#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4UserSteppingAction.hh"

class Opticks ; 

class G4Run ; 
class G4Event ; 
class G4Track ; 
class G4Step ; 

struct AManager
    : 
    public G4UserRunAction,
    public G4UserEventAction,
    public G4UserTrackingAction,
    public G4UserSteppingAction,
    public CManager
{
    AManager(Opticks* ok); 
    virtual ~AManager(); 
};

