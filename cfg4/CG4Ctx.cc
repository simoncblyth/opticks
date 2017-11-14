#include <sstream>

#include "G4OpticalPhoton.hh"
#include "G4Track.hh"
#include "G4Event.hh"

#include "CTrack.hh"
#include "CG4Ctx.hh"

#include "PLOG.hh"


void CG4Ctx::init()
{
    _event = NULL ; 
    _event_id = -1 ; 

    _track = NULL ; 
    _track_id = -1 ; 
    _optical = false ; 
    _pdg_encoding = 0 ; 
    _parent_id = -1 ; 

    _photon_id = -1 ; 
    _record_id = -1 ; 
    _debug = false ; 
    _other = false ; 
    _reemtrack = false ; 
}

void CG4Ctx::setEvent(const G4Event* event)
{
    _event = const_cast<G4Event*>(event) ; 
    _event_id = event->GetEventID() ;
}

void CG4Ctx::setTrack(const G4Track* track)
{
    G4ParticleDefinition* particle = track->GetDefinition();

    _track = const_cast<G4Track*>(track) ; 
    _track_id = CTrack::Id(track) ;
    _parent_id = CTrack::ParentId(track) ;
    _optical = particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    _pdg_encoding = particle->GetPDGEncoding();

    if(_optical) setTrackOptical();
}

void CG4Ctx::setTrackOptical()
{
     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ; 

     _track->UseGivenVelocity(true);

     // NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
     //    are trumpled by G4Track::CalculateVelocity 

     _primary_id = CTrack::PrimaryPhotonID(_track) ;    // layed down in trackinfo by custom Scintillation process
     _photon_id = _primary_id >= 0 ? _primary_id : _track_id ; 
     _reemtrack = _primary_id >= 0 ? true        : false ; 

     // retaining original photon_id from prior to reemission effects the continuation

    _record_id = _photons_per_g4event*_event_id + _photon_id ; 

}

std::string CG4Ctx::desc() const 
{
    std::stringstream ss ; 
    ss 
        << ( _debug ? " --dindex " : "" )
        << ( _other ? " --oindex " : "" )
        << " record_id " << _record_id
        << " event_id " << _event_id
        << " track_id " << _track_id
        << " photon_id " << _photon_id
        << " parent_id " << _parent_id
        << " primary_id " << _primary_id
        << " reemtrack " << _reemtrack
        ;
    return ss.str();
}



