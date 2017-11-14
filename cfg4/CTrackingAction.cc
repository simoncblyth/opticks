// g4-

#include "CFG4_PUSH.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"

#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"

// cg4-
#include "CG4Ctx.hh"
#include "CG4.hh"
#include "CRecorder.hh"
#include "CTrack.hh"
#include "CSteppingAction.hh"
#include "CTrackingAction.hh"

#include "PLOG.hh"


/**
CTrackingAction
=================

Canonical instance (m_ta) is ctor resident of CG4 

**/

CTrackingAction::CTrackingAction(CG4* g4)
   : 
   G4UserTrackingAction(),
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_recorder(g4->getRecorder()),
   m_sa(g4->getSteppingAction()),

   //m_event(NULL),
   //m_event_id(-1),

   //m_track(NULL),
   //m_track_id(-1),
   //m_parent_id(-1),
   m_track_status(fAlive),
   //m_particle(NULL),
   //m_pdg_encoding(0),
   //m_optical(false),
   //m_reemtrack(false),
   m_dump(false)
   //m_primary_id(-1),
   //m_photon_id(-1),
   //m_photons_per_g4event(0)
   //m_record_id(-1),
   //m_debug(false),
   //m_other(false)
{ 
}


CTrackingAction::~CTrackingAction()
{ 
}

void CTrackingAction::initEvent(OpticksEvent* evt)
{
   // invoked from CG4::initEvent
    m_ctx._photons_per_g4event = evt->getNumPhotonsPerG4Event() ; 

    LOG(info) << "CTrackingAction::initEvent"
              << " photons_per_g4event " << m_ctx._photons_per_g4event
              ;
}

void CTrackingAction::setEvent()
{
    m_sa->setEvent();
}

void CTrackingAction::setTrack(const G4Track* track)
{
    m_ctx.setTrack(track);

    m_track_status = track->GetTrackStatus(); 
 
    m_ctx._debug = m_ok->isDbgPhoton(m_ctx._record_id) ; // from option: --dindex=1,100,1000,10000 
    m_ctx._other = m_ok->isOtherPhoton(m_ctx._record_id) ; // from option: --oindex=1,100,1000,10000 
    m_dump = m_ctx._debug || m_ctx._other ; 

    m_sa->setTrack();
}



//void CTrackingAction::setRecordId(int record_id )
//{
//   // m_sa->setRecordId(record_id, _debug, other);
//}



void CTrackingAction::dump(const char* msg )
{
    LOG(info) << "." ;
    LOG(info) << msg  
              << " ctx " << m_ctx.desc()
               ; 
}

void CTrackingAction::postinitialize()
{
    assert(m_ctx._track_id == -1);
    assert(m_ctx._parent_id == -1);
    LOG(trace) << "CTrackingAction::postinitialize" 
              << brief()
               ;
}

std::string CTrackingAction::brief()
{
    std::stringstream ss ; 
    ss  
       << " track_id " << m_ctx._track_id
       << " parent_id " << m_ctx._parent_id
       ;
    return ss.str();
}


void CTrackingAction::PreUserTrackingAction(const G4Track* track)
{
   // TODO: move to CEventAction
   // const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;
   // setEvent(event);

    setTrack(track);

    LOG(trace) << "CTrackingAction::PreUserTrackingAction"
              << brief()  
               ;
}

void CTrackingAction::PostUserTrackingAction(const G4Track* track)
{
    int track_id = CTrack::Id(track) ;
    assert( track_id == m_ctx._track_id );
    assert( track == m_ctx._track );

    LOG(trace) << "CTrackingAction::PostUserTrackingAction" 
              << brief() 
              ;

    if(m_ctx._optical)
    {
        m_recorder->posttrack();
    } 
}


