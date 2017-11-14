#pragma once

#include "CFG4_API_EXPORT.hh"
#include <string>

class G4Event ; 
class G4Track ; 

/**
CG4Ctx
=======

Context shared between the CEventAction, CTrackingAction, CSteppingAction
and CRecorder


**/

struct CFG4_API CG4Ctx
{
    int  _photons_per_g4event ;

    G4Event* _event ; 
    int  _event_id ;

    // CTrackingAction::setTrack
    G4Track* _track ; 
    int  _track_id ;
    int  _parent_id ;
    bool _optical ; 
    int  _pdg_encoding ;

    // CTrackingAction::setPhotonId    
    int  _primary_id ; // used for reem continuation 
    int  _photon_id ;
    bool _reemtrack ; 

    // CTrackingAction::setRecordId
    int  _record_id ;
    bool _debug ; 
    bool _other ; 

    void init();
    void setEvent(const G4Event* event);
    void setTrack(const G4Track* track);
    void setTrackOptical();
    std::string desc() const  ; 

}; 


