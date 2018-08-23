#include <string>

#include "G4Event.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4HCtable.hh"

#include "EventAction.hh"
#include "SensitiveDetector.hh"
#include "OpHit.hh"

#ifdef WITH_OPTICKS
#include "G4Opticks.hh"
#include "NPY.hpp"
#endif

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

#ifdef WITH_OPTICKS
    G4Opticks* ok = G4Opticks::GetOpticks() ;
    int num_hits = ok->propagateOpticalPhotons() ;  
    NPY<float>* hits = ok->getHits(); 
    assert( hits->getNumItems() == unsigned(num_hits) ) ; 
    LOG(error) << " num_hits " << num_hits ; 
#endif

    //addDummyHits(HCE);
    SensitiveDetector::DumpHitCollections(HCE);

    // A possible alternative location to invoke the GPU propagation
    // and add hits in bulk to hit collections would be SensitiveDetector::EndOfEvent  
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



