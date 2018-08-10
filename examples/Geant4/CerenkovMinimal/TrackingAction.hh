#pragma once

#include "G4UserTrackingAction.hh"

class G4Track ; 
struct Ctx ; 

struct TrackingAction : public G4UserTrackingAction
{
    TrackingAction(Ctx* ctx_); 
    virtual void PreUserTrackingAction(const G4Track* track);
    virtual void PostUserTrackingAction(const G4Track* track);
    Ctx*  ctx ; 
};
