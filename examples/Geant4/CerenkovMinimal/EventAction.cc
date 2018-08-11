#include <string>

#include "G4Event.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4HCtable.hh"

#include "EventAction.hh"
#include "SensitiveDetector.hh"
#include "OpHit.hh"

#include "Ctx.hh"
#include "PLOG.hh"

EventAction::EventAction(Ctx* ctx_)
    :
    ctx(ctx_)
{
}

void EventAction::BeginOfEventAction(const G4Event* anEvent)
{
    ctx->setEvent(anEvent); 
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    G4HCofThisEvent* HCE = event->GetHCofThisEvent() ;
    assert(HCE); 

    addDummyHits(HCE);
    SensitiveDetector::DumpHitCollections(HCE);
}

void EventAction::addDummyHits(G4HCofThisEvent* HCE)
{
    OpHitCollection* A = SensitiveDetector::GetHitCollection(HCE, "SD0/OpHitCollectionA");
    OpHitCollection* B = SensitiveDetector::GetHitCollection(HCE, "SD0/OpHitCollectionB");
    for(unsigned i=0 ; i < 10 ; i++) 
    {
        OpHitCollection* HC = i % 2 == 0 ? A : B ; 
        HC->insert( OpHit::MakeDummyHit() ) ;
    }
}



