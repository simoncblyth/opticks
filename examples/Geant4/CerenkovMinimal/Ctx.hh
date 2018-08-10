#pragma once

/**
Follow pattern of CG4Ctx
**/

#include <string>
#include "G4ThreeVector.hh"

class G4Event ;
class G4Track ;
class G4Step ;

struct Ctx 
{
    const G4Event*  _event ; 
    int             _event_id ; 

    const G4Track*  _track ; 
    int             _track_id ; 
    int             _track_step_count ;
    int             _track_pdg_encoding ;
    bool            _track_optical ;  
    std::string     _track_particle_name ; 

    const G4Step*   _step ;  
    int             _step_id ;
    G4ThreeVector   _step_origin ;

    void setEvent(const G4Event* event);
    void setTrack(const G4Track* track);
    void setStep(const G4Step* step);
};





