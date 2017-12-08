#include <sstream>

#include "G4OpticalPhoton.hh"
#include "G4Track.hh"
#include "G4Event.hh"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"
#include "Opticks.hh"

#include "CProcessManager.hh"
#include "CTrack.hh"
#include "CG4Ctx.hh"

#include "PLOG.hh"


CG4Ctx::CG4Ctx(Opticks* ok)
    :
    _ok(ok),
    _pindex(ok->getPrintIndex(0)),
    _print(false)
{
    init();

    _dbgrec = ok->isDbgRec() ;   // machinery debugging 
    _dbgseq = ok->getDbgSeqhis() || ok->getDbgSeqmat() ;  // content debugging 
    _dbgzero = ok->isDbgZero() ; 
}

bool CG4Ctx::is_dbg() const 
{
    return _dbgrec || _dbgseq || _dbgzero ;
}


unsigned CG4Ctx::step_limit() const 
{
    // *step_limit* is used by CRec::addStp (recstp) the "canned" step collection approach, 
    // which just collects steps and makes sense of them later...
    // This has the disadvantage of needing to collect StepTooSmall steps (eg from BR turnaround)  
    // that are subsequently thrown : this results in the stem limit needing to 
    // be twice the size you might expect to handle hall-of-mirrors tboolean-truncate.
    assert( _ok_event_init ); 
    return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
}

unsigned CG4Ctx::point_limit() const 
{
    // *point_limit* is used by CRec::addPoi (recpoi) the "live" point collection approach, 
    // which makes sense of the points as they arrive, 
    // this has advantage of only storing the needed points. 
    //
    // DO NOT ADD +1 LEEWAY HERE : OTHERWISE TRUNCATION BEHAVIOUR IS CHANGED
    // see notes/issues/cfg4-point-recording.rst
    //
    assert( _ok_event_init ); 
    return ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
}



void CG4Ctx::init()
{
    _dbgrec = false ; 
    _dbgseq = false ; 
    _dbgzero = false ; 

    _photons_per_g4event = 0 ; 
    _steps_per_photon = 0 ; 
    _gen = 0 ; 
    _record_max = 0 ;
    _bounce_max = 0 ; 

    _ok_event_init = false ; 
    _event = NULL ; 
    _event_id = -1 ; 
    _event_total = 0 ; 
    _event_track_count = 0 ; 

    _track = NULL ; 
    _process_manager = NULL ; 
    _track_id = -1 ; 
    _track_total = 0 ;
    _track_step_count = 0 ;  

    _parent_id = -1 ; 
    _optical = false ; 
    _pdg_encoding = 0 ; 

    _primary_id = -1 ; 
    _photon_id = -1 ; 
    _record_id = -1 ; 
    _reemtrack = false ; 

    _rejoin_count = 0 ; 
    _primarystep_count = 0 ; 
    _stage = CStage::UNKNOWN ; 

    _record_id = -1 ; 
    _debug = false ; 
    _other = false ; 
    _dump = false ; 
    _dump_count = 0 ; 


    _step = NULL ; 
    _step_id = -1 ;
    _step_total = 0 ; 
 
}



std::string CG4Ctx::desc_stats() const 
{
    std::stringstream ss ; 
    ss << "CG4Ctx::desc_stats"
       << " dump_count " << _dump_count  
       << " event_total " << _event_total 
       << " event_track_count " << _event_track_count
       ;
    return ss.str();
}

void CG4Ctx::initEvent(const OpticksEvent* evt)
{
    _ok_event_init = true ;
    _photons_per_g4event = evt->getNumPhotonsPerG4Event() ; 
    _steps_per_photon = evt->getMaxRec() ;    
    _record_max = evt->getNumPhotons();   // from the genstep summation
    _bounce_max = evt->getBounceMax();

    const char* typ = evt->getTyp();
    _gen = OpticksFlags::SourceCode(typ);
    assert( _gen == TORCH || _gen == G4GUN  );
  
    LOG(info) << "CG4Ctx::initEvent"
              << " photons_per_g4event " << _photons_per_g4event
              << " steps_per_photon " << _steps_per_photon
              << " gen " << _gen
              ;
}

std::string CG4Ctx::desc_event() const 
{
    std::stringstream ss ; 
    ss << "CG4Ctx::desc_event" 
       << " photons_per_g4event " << _photons_per_g4event
       << " steps_per_photon " << _steps_per_photon
       << " record_max " << _record_max
       << " bounce_max " << _bounce_max
       << " _gen " << _gen
       ;
    return ss.str();
}







void CG4Ctx::setEvent(const G4Event* event) // invoked by CEventAction::setEvent
{
    _event = const_cast<G4Event*>(event) ; 
    _event_id = event->GetEventID() ;

    _event_total += 1 ; 
    _event_track_count = 0 ; 
}

void CG4Ctx::setTrack(const G4Track* track) // invoked by CTrackingAction::setTrack
{
    G4ParticleDefinition* particle = track->GetDefinition();

    _track = const_cast<G4Track*>(track) ; 
    _track_id = CTrack::Id(track) ;

    _process_manager = CProcessManager::Current(_track);

    _track_step_count = 0 ; 
    _event_track_count += 1 ; 
    _track_total += 1 ;
    
    _parent_id = CTrack::ParentId(track) ;
    _optical = particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    _pdg_encoding = particle->GetPDGEncoding();


    _step = NULL ; 
    _step_id = -1 ; 

    if(_optical) setTrackOptical();
}




void CG4Ctx::setStep(const G4Step* step) // invoked by CSteppingAction::setStep
{
    _step = const_cast<G4Step*>(step) ; 
    _step_id = CTrack::StepId(_track);
    _step_total += 1 ; 
    _track_step_count += 1 ; 

    if(_step_id == 0)
    {
        const G4StepPoint* pre = _step->GetPreStepPoint() ;
        _step_origin = pre->GetPosition();
    }

    if(_optical) setStepOptical();
}

void CG4Ctx::setTrackOptical() // invoked by CG4Ctx::setTrack
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
    _record_fraction = double(_record_id)/double(_record_max) ;  

    // moved from CTrackingAction::setTrack
    _debug = _ok->isDbgPhoton(_record_id) ; // from option: --dindex=1,100,1000,10000 
    _other = _ok->isOtherPhoton(_record_id) ; // from option: --oindex=1,100,1000,10000 
    _dump = _debug || _other ; 

    _print = _pindex > -1 && _record_id == _pindex ; 


    if(_dump) _dump_count += 1 ; 


    // moved from  CSteppingAction::setPhotonId
    // essential for clearing counts otherwise, photon steps never cleared 
    _rejoin_count = 0 ; 
    _primarystep_count = 0 ; 


}

void CG4Ctx::setStepOptical() // invoked by CG4Ctx::setStep
{
    if( !_reemtrack )     // primary photon, ie not downstream from reemission 
    {
        _stage = _primarystep_count == 0  ? CStage::START : CStage::COLLECT ;
        _primarystep_count++ ; 
    } 
    else 
    {
        _stage = _rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;   
        _rejoin_count++ ; 
        // rejoin count is zeroed in setTrackOptical, so each remission generation trk will result in REJOIN 
    }
}


std::string CG4Ctx::desc_step() const 
{
    G4TrackStatus track_status = _track->GetTrackStatus(); 

    std::stringstream ss ; 
    ss << "CG4Ctx::desc_step" 
       << " step_total " << _step_total
       << " event_id " << _event_id
       << " track_id " << _track_id
       << " track_step_count " << _track_step_count
       << " step_id " << _step_id
       << " trackStatus " << CTrack::TrackStatusString(track_status)
       ;

    return ss.str();
}


std::string CG4Ctx::brief() const 
{
    std::stringstream ss ; 
    ss 
        << " record_id " << _record_id
        ;
 
    return ss.str();
}
 

std::string CG4Ctx::desc() const 
{
    std::stringstream ss ; 
    ss 
        << ( _dbgrec ? " [--dbgrec] " : "" )
        << ( _dbgseq ? " [--dbgseqmat 0x.../--dbgseqhis 0x...] " : "" )
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



