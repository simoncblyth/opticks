#pragma once

#include "CFG4_API_EXPORT.hh"
#include "G4ThreeVector.hh"
#include "CStage.hh"
#include <string>



class G4Event ; 
class G4Track ; 
class G4Step ; 

class OpticksEvent ; 
class Opticks ; 

/**
CG4Ctx
=======

Context shared between the CEventAction, CTrackingAction, CSteppingAction
and CRecorder


**/

struct CFG4_API CG4Ctx
{
    Opticks* _ok ; 

    // CG4::init
    bool _dbgrec ; 
    bool _dbgseq ; 
    bool _dbgzero ; 

    // CG4::initEvent
    int  _photons_per_g4event ;
    unsigned  _steps_per_photon  ;
    unsigned  _gen  ;
    unsigned  _record_max ; 
    unsigned  _bounce_max ; 


    // CG4Ctx::setEvent
    bool  _ok_event_init ; 
    G4Event* _event ; 
    int  _event_id ;
    int  _event_total ; 
    int  _event_track_count ; 

    // CG4Ctx::setTrack
    G4Track* _track ; 
    int  _track_id ;
    int  _track_total ; 
    int  _track_step_count ; 
    int  _parent_id ;
    bool _optical ; 
    int  _pdg_encoding ;

    // CG4Ctx::setTrackOptical
    int  _primary_id ; // used for reem continuation 
    int  _photon_id ;
    bool _reemtrack ; 
    int  _record_id ;
    float _record_fraction ; // used with --reflectcheat
    // zeroed in CG4Ctx::setTrackOptical incremented in CG4Ctx::setStep
    int  _rejoin_count ; 
    int  _primarystep_count ; 
    CStage::CStage_t  _stage ; 

    // CTrackingAction::setTrack
    bool _debug ; 
    bool _other ; 
    bool _dump ; 
    int  _dump_count ; 

    // m_ctx.setStep invoked from CSteppingAction::setStep
    G4Step* _step ; 
    int _step_id ; 
    int _step_total ;
    G4ThreeVector _step_origin ; 
 

    CG4Ctx(Opticks* ok);

    void init();
    void initEvent(const OpticksEvent* evt);

    void setEvent(const G4Event* event);
    void setTrack(const G4Track* track);
    void setTrackOptical();
    void setStep(const G4Step* step);
    void setStepOptical();
    unsigned  step_limit() const ; 
    unsigned  point_limit() const ; 
    bool      is_dbg() const ; 

    std::string desc_event() const ;
    std::string desc_step() const ;
    std::string desc() const  ; 
    std::string brief() const  ; 
    std::string desc_stats() const ;

}; 


