#include "TrackingAction.hh"
#include "Ctx.hh"

TrackingAction::TrackingAction(Ctx* ctx_)
    :
    ctx(ctx_)
{
}

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    ctx->setTrack(track); 
}
void TrackingAction::PostUserTrackingAction(const G4Track* track)
{
    ctx->postTrack(track);     
}


