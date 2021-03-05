/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
#include "G4OpticksHit.hh"
#include "OpticksFlags.hh"
#endif

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

void EventAction::EndOfEventAction(const G4Event* event)
{
    G4HCofThisEvent* HCE = event->GetHCofThisEvent() ;
    assert(HCE); 

#ifdef WITH_OPTICKS
    G4cout << "\n###[ EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons\n" << G4endl ; 

    G4Opticks* g4ok = G4Opticks::Get() ;
    G4int eventID = event->GetEventID() ; 
    int num_hits = g4ok->propagateOpticalPhotons(eventID) ;  

    G4cout 
           << "EventAction::EndOfEventAction"
           << " eventID " << eventID
           << " num_hits " << num_hits 
           << G4endl 
           ; 

    G4OpticksHit hit ;
    G4OpticksHitExtra* hit_extra = NULL ;

    for(unsigned i=0 ; i < num_hits ; i++)
    {   
        g4ok->getHit(i, &hit, hit_extra ); 
        std::cout 
            << std::setw(5) << i 
            << " boundary "           << std::setw(4) << hit.boundary 
            << " sensorIndex "        << std::setw(5) << hit.sensorIndex 
            << " nodeIndex "          << std::setw(5) << hit.nodeIndex 
            << " photonIndex "        << std::setw(5) << hit.photonIndex 
            << " flag_mask    "       << std::setw(10) << std::hex << hit.flag_mask  << std::dec
            << " sensor_identifier "  << std::setw(10) << std::hex << hit.sensor_identifier << std::dec
            << " wavelength "         << std::setw(8) << hit.wavelength 
            << " time "               << std::setw(8) << hit.time
            << " global_position "    << hit.global_position 
            << " " << OpticksFlags::FlagMask(hit.flag_mask, true)
            << std::endl 
            ;
    }

    g4ok->reset();  // necessary to prevent gensteps keeping to accumulate

    G4cout << "\n###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons\n" << G4endl ; 
#endif

    //addDummyHits(HCE);
    G4cout 
         << "EventAction::EndOfEventAction"
         << " DumpHitCollections "
         << G4endl 
         ; 
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



