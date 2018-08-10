#include "EventAction.hh"
#include "Ctx.hh"

EventAction::EventAction(Ctx* ctx_)
    :
    ctx(ctx_)
{
}

void EventAction::BeginOfEventAction(const G4Event* anEvent)
{
    ctx->setEvent(anEvent); 
}
void EventAction::EndOfEventAction(const G4Event* )
{
}


