#include "SteppingAction.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "Ctx.hh"

SteppingAction::SteppingAction(Ctx* ctx_)
    :
    ctx(ctx_)
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    ctx->setStep(step); 
}



