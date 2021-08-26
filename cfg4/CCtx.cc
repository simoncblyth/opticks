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

#include "G4OpticalPhoton.hh"
#include "G4Track.hh"
#include "G4Event.hh"

#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"
#include "OpticksEvent.hh"
#include "Opticks.hh"

#include "GGeo.hh"

#include "CGenstep.hh"
#include "CGenstepCollector.hh"
#include "CBoundaryProcess.hh"
#include "CProcessManager.hh"
#include "CEvent.hh"
#include "CTrack.hh"
#include "CCtx.hh"
#include "CEventInfo.hh"
#include "CPhotonInfo.hh"

#include "PLOG.hh"


const plog::Severity CCtx::LEVEL = PLOG::EnvLevel("CCtx", "DEBUG") ; 

const unsigned CCtx::CK = OpticksGenstep::SourceCode("G4Cerenkov_1042");   
const unsigned CCtx::SI = OpticksGenstep::SourceCode("G4Scintillation_1042"); 
const unsigned CCtx::TO = OpticksGenstep::SourceCode("fabricated");   


CCtx::CCtx(Opticks* ok)
    :
    _ok(ok),
    _mode(ok->getManagerMode()),
    _pindex(ok->getPrintIndex(0)),
    _print(false),
    _gsc(nullptr),       // defer to setEvent
    _genstep_index(-1)
{
    init();

    _dbgrec = ok->isDbgRec() ;   // machinery debugging   --dbgrec 
    _dbgseq = ok->getDbgSeqhis() || ok->getDbgSeqmat() ;  // history sequence debugging   --dbgseqhis 0x...  --dbgseqmat 0x...
    _dbgzero = ok->isDbgZero() ;   // --dbgzero 
}


Opticks*  CCtx::getOpticks() const
{
    return _ok ; 
}


bool CCtx::is_dbg() const   // commandline includes : --dbgrec --dbgseqhis 0x.. --dbgseqmat 0x..  --dbgzero
{
    unsigned numDbgPhoton = _ok->getNumDbgPhoton() ;       // --dindex 1,2,3
    unsigned numOtherPhoton = _ok->getNumOtherPhoton() ;   // --oindex 4,5,6
    return _dbgrec || _dbgseq || _dbgzero || ( numDbgPhoton > 0 ) || ( numOtherPhoton > 0 ) ;   
}


/**
CCtx::step_limit
-------------------

*step_limit* is used by CRec::addStp (recstp) the "canned" step collection approach, 
which just collects steps and makes sense of them later...
This has the disadvantage of needing to collect StepTooSmall steps (eg from BR turnaround)  
that are subsequently thrown : this results in the step limit needing to 
be twice the size you might expect to handle hall-of-mirrors tboolean-truncate.

This also has disadvantage that tail consumption as checked with "--utaildebug" 
does not match between Opticks and Geant4, see 

* notes/issues/ts-box-utaildebug-decouple-maligned-from-deviant.rst


**/

unsigned CCtx::step_limit() const 
{
    assert( _ok_event_init ); 
    return 1 + 2*( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
}

/**
CCtx::point_limit
---------------------

Returns the larger of the below:

_steps_per_photon
     number of photon step points to record into record array
_bounce_max
     number of bounces before truncation, often 1 less than _steps_per_photon but need not be  

*point_limit* is used by CRec::addPoi (recpoi) the "live" point collection approach, 
which makes sense of the points as they arrive, 
this has advantage of only storing the needed points. 

DO NOT ADD +1 LEEWAY HERE : OTHERWISE TRUNCATION BEHAVIOUR IS CHANGED
see notes/issues/cfg4-point-recording.rst

**/

unsigned CCtx::point_limit() const 
{
    assert( _ok_event_init ); 
    return ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
    //return _bounce_max  ;
}



void CCtx::init()
{
    _dbgrec = false ; 
    _dbgseq = false ; 
    _dbgzero = false ; 

    _photons_per_g4event = 0 ; 
    _steps_per_photon = 0 ; 
    _record_max = 0 ;
    _bounce_max = 0 ; 

    _ok_event_init = false ; 
    _event = NULL ; 
    _event_id = -1 ; 
    _event_total = 0 ; 
    _event_track_count = 0 ; 

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
    _mask_index = -1 ; 

    _rejoin_count = 0 ; 
    _primarystep_count = 0 ; 
    _stage = CStage::UNKNOWN ; 

    _debug = false ; 
    _other = false ; 
    _dump = false ; 
    _dump_count = 0 ; 


    _step = NULL ; 
    _noZeroSteps = -1 ; 
    _step_id = -1 ;
    _step_total = 0 ; 

#ifdef USE_CUSTOM_BOUNDARY
    _boundary_status = Ds::Undefined ; 
    _prior_boundary_status = Ds::Undefined ; 
#else
    _boundary_status = Undefined ; 
    _prior_boundary_status = Undefined ; 
#endif

}


std::string CCtx::desc_stats() const 
{
    std::stringstream ss ; 
    ss << "CCtx::desc_stats"
       << " dump_count " << _dump_count  
       << " event_total " << _event_total 
       << " event_track_count " << _event_track_count
       ;
    return ss.str();
}

/**
CCtx::initEvent
--------------------

Collect the parameters of the OpticksEvent which 
dictate what needs to be collected.

Invoked by 

**/

void CCtx::initEvent(const OpticksEvent* evt)
{
    _ok_event_init = true ;
    _photons_per_g4event = evt->getNumPhotonsPerG4Event() ; 
    _steps_per_photon = evt->getMaxRec() ;   // number of points to be recorded into record buffer   
    _record_max = evt->getNumPhotons();      // from the genstep summation, hmm with dynamic running this will start as zero 

    _bounce_max = evt->getBounceMax();       // maximum bounce allowed before truncation will often be 1 less than _steps_per_photon but need not be 
    unsigned bounce_max_2 = evt->getMaxBounce();    
    assert( _bounce_max == bounce_max_2 ) ; // TODO: eliminate or rename one of those

    const char* typ = evt->getTyp();

    LOG(LEVEL)
        << " _record_max (numPhotons from genstep summation) " << _record_max 
        << " photons_per_g4event " << _photons_per_g4event
        << " _steps_per_photon (maxrec) " << _steps_per_photon
        << " _bounce_max " << _bounce_max
        << " typ " << typ
        ;

}

std::string CCtx::desc_event() const 
{
    std::stringstream ss ; 
    ss << "CCtx::desc_event" 
       << " photons_per_g4event " << _photons_per_g4event
       << " steps_per_photon " << _steps_per_photon
       << " record_max " << _record_max
       << " bounce_max " << _bounce_max
       << " _gs.desc " << _gs.desc()
       ;
    return ss.str();
}


/**
CCtx::setEvent
-----------------

Invoked by CManager::BeginOfEventAction

The G4Event primaries are examined to check for input photons 
which result in setting _number_of_input_photons 

When _number_of_input_photons is greater than 0 
CManager "mocks" a genstep by calling CCtx::setGenstep
in normal running that gets called from CManager::BeginOfGenstep

**/

void CCtx::setEvent(const G4Event* event) 
{
    _gsc = CGenstepCollector::Get() ; 

    _event = const_cast<G4Event*>(event) ; 
    _event_id = event->GetEventID() ;

    _event_total += 1 ; 
    _event_track_count = 0 ; 
    _track_optical_count = 0 ; 

    _genstep_index = -1 ; 

    _number_of_input_photons = CEvent::NumberOfInputPhotons(event); 
    LOG(LEVEL) 
        << "_number_of_input_photons " << _number_of_input_photons 
        << "_genstep_index " << _genstep_index
        ; 
}


unsigned CCtx::getNumPhotons() const
{
    return _gsc ? _gsc->getNumPhotons() : 0 ; 

}


/**
CCtx::BeginOfGenstep
------------------------

Invoked by CGenstepCollector::addGenstep from the CGenstepCollector::collect methods 

**/

void CCtx::BeginOfGenstep(unsigned genstep_index, char gentype, int num_photons, int offset )
{
    setGenstep(genstep_index, gentype, num_photons, offset); 
}

void CCtx::setGenstep(unsigned genstep_index, char gentype, int num_photons, int offset)
{
    _genstep_index += 1 ;    // _genstep_index starts at -1 and is reset to -1 by CCtx::setEvent, so it becomes a zero based event local index
    _genstep_num_photons = num_photons ; 

    bool genstep_index_match = int(genstep_index) == _genstep_index ;

    if(!genstep_index_match) 
        LOG(fatal)
            << " genstep_index(argument) " << genstep_index
            << " _genstep_index(counter) " << _genstep_index
            << " gentype " << gentype
            << " _gentype " << _gentype
            ;
        
    //assert( genstep_index_match ); 

    setGentype(gentype); 
}

void CCtx::setGentype(char gentype)
{
    _gentype = gentype ; 
}

unsigned CCtx::getGenflag() const
{
    return OpticksGenstep::GentypeToPhotonFlag(_gentype); 
}


/**
CCtx::setTrack
------------------

Invoked by CManager::PreUserTrackingAction

**/

void CCtx::setTrack(const G4Track* track) 
{
    G4ParticleDefinition* particle = track->GetDefinition();

    _nidx = -1 ;   // gets set at postTrack
    _track = track ; 

    G4Track* mtrack = const_cast<G4Track*>(track) ; 
    _process_manager = CProcessManager::Current(mtrack);

    _track_status = track->GetTrackStatus(); 
    _track_id = CTrack::Id(track) ;    // 0-based id 

    _track_step_count = 0 ; 
    _event_track_count += 1 ; 
    _track_total += 1 ;
    
    _parent_id = CTrack::ParentId(track) ;
    _optical = particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    _pdg_encoding = particle->GetPDGEncoding();

    _step = NULL ; 
    _step_id = -1 ; 
    _step_id_valid = -1 ; 

    _hitflags = 0 ; 


    LOG(LEVEL) 
        << " _track_id " << _track_id
        << " track.GetGlobalTime " << track->GetGlobalTime()
        << " _parent_id " << _parent_id
        << " _pdg_encoding " << _pdg_encoding
        << " _optical " << _optical 
        << " _process_manager " << CProcessManager::Desc(_process_manager)
        ;

    if(_optical) setTrackOptical(mtrack);
}

/**
CCtx::postTrack
-----------------

**/

void CCtx::postTrack()
{
    const G4VPhysicalVolume* pv = _track->GetVolume() ; 
    int origin_copyNumber = pv->GetCopyNo() ;   // from G4PVPlacement subclass
    const void* origin = (void*)pv ; 

    _nidx = GGeo::Get()->findNodeIndex(origin, origin_copyNumber); 
} 



/**
CCtx::setTrackOptical
--------------------------

Invoked by CCtx::setTrack


UseGivenVelocity(true)
~~~~~~~~~~~~~~~~~~~~~~~~

NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
are trumpled by G4Track::CalculateVelocity 

**/

void CCtx::setTrackOptical(G4Track* mtrack) 
{
    mtrack->UseGivenVelocity(true);

    bool fabricate_unlabelled = true ; 
    _pho = CPhotonInfo::Get(mtrack, fabricate_unlabelled); 

    int pho_id = _pho.get_id();
    assert( pho_id > -1 ); 

    _gs = _gsc->getGenstep(_pho.gs) ; 
    assert( _gs.index == _pho.gs ); 

    _photon_id = pho_id ; // 0-based, absolute photon index within the event 
    _record_id = pho_id ; // used by CRecorder/CWriter is now absolute, following abandonment of onestep mode  
    _record_fraction = double(_record_id)/double(_record_max) ;  

    _track_optical_count += 1 ;   // CAREFUL : DOES NOT ACCOUNT FOR RE-JOIN 

    assert( _record_id > -1 ); 


    if(_number_of_input_photons > 0 && _record_id > _number_of_input_photons)
    {
        LOG(fatal)
            << " _number_of_input_photons " << _number_of_input_photons
            << " _photon_id " << _photon_id
            << " _record_id " << _record_id
            << " _parent_id " << _parent_id
            << " _pho " << _pho.desc()
            << " _gs " << _gs.desc()
            << " _track_optical_count " << _track_optical_count 
            ;
    }



    _mask_index = _ok->hasMask() ?_ok->getMaskIndex( _primary_id ) : -1 ;   // "original" index 

    _debug = _ok->isDbgPhoton(_photon_id) ; // from option: --dindex=1,100,1000,10000 
    _other = _ok->isOtherPhoton(_photon_id) ; // from option: --oindex=1,100,1000,10000 
    _dump = _debug || _other ; 

    _print = _pindex > -1 && _photon_id == _pindex ; 

    if(_dump) _dump_count += 1 ; 

    LOG(LEVEL) 
         << " _pho " << _pho.desc()  
         << " _photon_id " << _photon_id  
         << " _record_id " << _record_id  
         << " _pho.gn " << _pho.gn
         << " mtrack.GetGlobalTime " << mtrack->GetGlobalTime()
         << " _debug " << _debug
         << " _other " << _other
         << " _dump " << _dump
         << " _print " << _print
         << " _dump_count " << _dump_count
         ;

    _rejoin_count = 0 ; 
    _primarystep_count = 0 ; 
}



/**
CCtx::setStep
----------------

Invoked by CManager::UserSteppingAction

**/

void CCtx::setStep(const G4Step* step, int noZeroSteps) 
{
    _step = const_cast<G4Step*>(step) ; 
    _noZeroSteps = noZeroSteps ;  
    _step_id = CTrack::StepId(_track);
    if(_noZeroSteps == 0) _step_id_valid += 1;

    _step_total += 1 ; 
    _track_step_count += 1 ; 

    const G4StepPoint* pre = _step->GetPreStepPoint() ;
    const G4StepPoint* post = _step->GetPostStepPoint() ;

    _step_pre_status = pre->GetStepStatus();
    _step_post_status = post->GetStepStatus();

    if(_step_id == 0)
    {
        _step_origin = pre->GetPosition();
    }

    if(_optical) setStepOptical();
}


/**
CCtx::setStepOptical
-----------------------

Invoked by CCtx::setStep , sets:

1. _stage to START/COLLECT/REJOIN/RECOLL depending on _pho.gn "photon generation index" and prior step counts
2. _boundary_status

**/

void CCtx::setStepOptical() 
{
    if( _pho.gn == 0 ) // photon generation zero is the primary photon, ie not downstream from reemission 
    {
        _stage = _primarystep_count == 0  ? CStage::START : CStage::COLLECT ;
        _primarystep_count++ ; 
    } 
    else   // (unsigned)_pho_gn 1,2,3,...  
    {
        _stage = _rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;   
        _rejoin_count++ ; 
        // rejoin count is zeroed in setTrackOptical, so each remission generation trk will result in REJOIN 
    }

    LOG(LEVEL) 
        << " _pho.gn " << _pho.gn
        << " _stage " << CStage::Label(_stage)
        << " _primarystep_count " << _primarystep_count
        << " _rejoin_count " << _rejoin_count
        ;

    _prior_boundary_status = _boundary_status ; 
    _boundary_status = CBoundaryProcess::GetOpBoundaryProcessStatus() ;

    LOG(LEVEL) 
        <<  " _prior_boundary_status " << std::setw(35) << CBoundaryProcess::OpBoundaryString(_prior_boundary_status)
        <<  " _boundary_status " << std::setw(35) << CBoundaryProcess::OpBoundaryString(_boundary_status)
        ;
}




std::string CCtx::desc_step() const 
{
    G4TrackStatus track_status = _track->GetTrackStatus(); 

    std::stringstream ss ; 
    ss << "CCtx::desc_step" 
       << " step_total " << _step_total
       << " event_id " << _event_id
       << " track_id " << _track_id
       << " track_step_count " << _track_step_count
       << " step_id " << _step_id
       << " trackStatus " << CTrack::TrackStatusString(track_status)
       ;

    return ss.str();
}


std::string CCtx::brief() const 
{
    std::stringstream ss ; 
    ss 
        << " record_id " << _record_id
        ;
 
    return ss.str();
}
 

std::string CCtx::desc() const 
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

/**
CCtx::ProcessHits
---------------------

Invoked by the chain::

    G4OpticksRecorder::ProcessHits
    CManager::ProcessHits

Adds EFFICIENCY_COLLECT or EFFICIENCY_CULL to the _hitflags, 
these _hitflags are zeroed in CCtx::setTrack 

**/
void CCtx::ProcessHits( const G4Step* step, bool efficiency_collect )
{
    const G4Track* track = step->GetTrack();    
    bool fabricate_unlabelled = false ;
    CPho hit = CPhotonInfo::Get(track, fabricate_unlabelled); 

    if(!hit.is_missing())  
    {
        if(!_pho.isEqual(hit))
        {
            LOG(fatal)
                << " _pho not equal to hit "
                << "  _pho.desc " << _pho.desc()
                << " hit.desc " << hit.desc()
                ;
            //assert(0); 
        }
    }

    if( efficiency_collect )
    {
        _hitflags |= EFFICIENCY_COLLECT ; 
    }
    else
    {
        _hitflags |= EFFICIENCY_CULL ; 
    }


    LOG(LEVEL) 
        << " hit " << hit.desc() 
        << " efficiency_collect " << efficiency_collect
        << " _hitflags " << OpticksFlags::FlagMask( _hitflags, false )
        ; 

}


