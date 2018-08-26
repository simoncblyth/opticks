#pragma once

/**
Follow pattern of CG4Ctx
**/

#include <string>
#include "G4ThreeVector.hh"

class G4Event ;
class G4Track ;
class G4Step ;
class G4StepPoint ;

struct Ctx 
{
    static std::string Format(const G4Step* step, const char* msg );
    static std::string Format(const G4StepPoint* point, const char* msg );
    static std::string Format(const G4ThreeVector& vec, const char* msg, unsigned int fwid);

    const G4Event*  _event ; 
    int             _event_id ; 

    const G4Track*  _track ; 
    int             _track_id ; 
    int             _record_id ; 

    int             _track_step_count ;
    int             _track_pdg_encoding ;
    bool            _track_optical ;  
    std::string     _track_particle_name ; 


    const G4Step*   _step ;  
    int             _step_id ;
    G4ThreeVector   _step_origin ;

    void setEvent(const G4Event* event);

    void setTrack(const G4Track* track);
    void postTrack(const G4Track* track);
    void setTrackOptical(const G4Track* track);
    void postTrackOptical(const G4Track* track);

    void setStep(const G4Step* step);
};





