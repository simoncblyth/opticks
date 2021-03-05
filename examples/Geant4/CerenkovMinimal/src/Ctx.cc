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


#include <sstream>
#include <iomanip>

#include "Ctx.hh"

//#define WITH_DUMP 1


#ifdef WITH_OPTICKS
#include "G4Opticks.hh"
#include "TrackInfo.hh"
#endif


#include "G4Event.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4OpticalPhoton.hh"

#include "G4ThreeVector.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"



void Ctx::setEvent(const G4Event* event)
{
    _event = event ; 
    _event_id = event->GetEventID() ; 
    //_event_total += 1 ;
    //_event_track_count = 0 ; 

    _track = NULL ; 
    _track_id = -1 ; 
    _record_id = -1 ; 
}

void Ctx::setTrack(const G4Track* track)
{
    _track = track ; 
    _track_id = track->GetTrackID() - 1 ;

    _step = NULL ; 
    _step_id = -1 ;  

    const G4DynamicParticle* dp = track->GetDynamicParticle() ; 
    assert( dp ) ;  
    const G4ParticleDefinition* particle = dp->GetParticleDefinition() ;
    assert( particle ) ;  

    _track_particle_name = particle->GetParticleName() ; 

    _track_optical = particle == G4OpticalPhoton::OpticalPhotonDefinition() ;

    _track_pdg_encoding = particle->GetPDGEncoding() ;

    if(_track_optical)
    {
        setTrackOptical(track);  
    }
    else
    {
#ifdef WITH_OPTICKS
        unsigned num_gs = G4Opticks::Get()->getNumGensteps() ; 
        unsigned max_gs = G4Opticks::Get()->getMaxGensteps() ; // default of zero means no limit  
        bool kill = max_gs > 0 && num_gs >= max_gs ;   

#ifdef WITH_DUMP
        G4cout 
            << "Ctx::setTrack"   
            << " _track_particle_name " << _track_particle_name 
            << " _track_id " << _track_id 
            << " _step_id " << _step_id 
            << " num_gs " << num_gs 
            << " max_gs " << max_gs 
            << " kill " << kill
            << G4endl
            ;  
#endif

        if(kill)
        {
            const_cast<G4Track*>(track)->SetTrackStatus(fStopAndKill);
        }

#endif
    }
}

void Ctx::postTrack( const G4Track* track)
{
    if(_track_optical)
    {
        postTrackOptical(track);  
    }
#ifdef WITH_DUMP
    else
    {
        G4cout 
            << "Ctx::postTrack"
            << " _track_particle_name : " << _track_particle_name
            << G4endl 
            ; 
    }
#endif
}


void Ctx::setTrackOptical(const G4Track* track)
{
    const_cast<G4Track*>(track)->UseGivenVelocity(true);

#ifdef WITH_OPTICKS
    TrackInfo* info=dynamic_cast<TrackInfo*>(track->GetUserInformation()); 
    assert(info) ; 
    _record_id = info->photon_record_id ;  
    //std::cout << "Ctx::setTrackOptical.setAlignIndex " << _record_id << std::endl ;
    G4Opticks::Get()->setAlignIndex(_record_id);
#endif
}

void Ctx::postTrackOptical(const G4Track* track)
{
#ifdef WITH_OPTICKS
    TrackInfo* info=dynamic_cast<TrackInfo*>(track->GetUserInformation()); 
    assert(info) ; 
    assert( _record_id == info->photon_record_id ) ;  
    //std::cout << "Ctx::postTrackOptical " << _record_id << std::endl ;
    G4Opticks::Get()->setAlignIndex(-1);
#endif
}



void Ctx::setStep(const G4Step* step)
{  

    assert( _track ) ; 

    
    const G4Track* track = _track ? _track : step->GetTrack() ; 
    // curious 10.4.2 getting setStep before setTrack ?

    _step = step ; 
    _step_id = track->GetCurrentStepNumber() - 1 ;

    _track_step_count += 1 ;
    
    const G4StepPoint* pre = _step->GetPreStepPoint() ;
    //const G4StepPoint* post = _step->GetPostStepPoint() ;

    if(_step_id == 0) _step_origin = pre->GetPosition();

#ifdef WITH_DUMP
    if(!_track_optical)
    {
        G4cout
            << "Ctx::setStep" 
            << " _step_id " << _step_id 
            << " num_gs " << G4Opticks::Get()->getNumGensteps() 
            << G4endl
            ;  
    }

#endif

}



std::string Ctx::Format(const G4ThreeVector& vec, const char* msg, unsigned int fwid)
{   
    std::stringstream ss ; 
    ss 
       << " " 
       << msg 
       << "[ "
       << std::fixed
       << std::setprecision(3)  
       << std::setw(fwid) << vec.x()
       << std::setw(fwid) << vec.y()
       << std::setw(fwid) << vec.z()
       << "] "
       ;
    return ss.str();
}

std::string Ctx::Format(const G4StepPoint* point, const char* msg )
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();
    std::stringstream ss ; 
    ss 
       << " " 
       << msg 
       << Format(pos, "pos", 10)
       << Format(dir, "dir", 10)
       << Format(pol, "pol", 10)
       ;
    return ss.str();
}

std::string Ctx::Format(const G4Step* step, const char* msg )
{
    const G4StepPoint* pre = step->GetPreStepPoint() ;
    const G4StepPoint* post = step->GetPostStepPoint() ;
    std::stringstream ss ; 
    ss 
       << " " 
       << msg 
       << std::endl 
       << Format(pre,  "pre ")
       << std::endl 
       << Format(post, "post")
       ; 
    return ss.str();
}




