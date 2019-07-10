// g4-

#include "CFG4_PUSH.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"

// cg4-
#include "CG4Ctx.hh"
#include "CG4.hh"
#include "CTrack.hh"
#include "CTrackingAction.hh"

#include "PLOG.hh"


const plog::Severity CTrackingAction::LEVEL = PLOG::EnvLevel("CTrackingAction", "DEBUG") ; 


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
   m_ok(g4->getOpticks())
{ 
}

CTrackingAction::~CTrackingAction()
{ 
}

void CTrackingAction::postinitialize()
{
    assert(m_ctx._track_id == -1);
    assert(m_ctx._parent_id == -1);
    LOG(verbose) << "CTrackingAction::postinitialize" << brief() ; 
}


/**
CTrackingAction::PreUserTrackingAction
-----------------------------------------

Invoked by G4TrackingManager::ProcessOneTrack immediately after:: 

   G4SteppingManager::SetInitialStep(G4Track* valueTrack)
   G4Step::InitializeStep( G4Track* aValue )
   
   g4-;g4-cls G4TrackingManager
   g4-;g4-cls G4SteppingManager
   g4-;g4-cls G4Step

**/

void CTrackingAction::PreUserTrackingAction(const G4Track* track)
{
    setTrack(track);

    LOG(LEVEL) << brief()  ;

    if(m_ctx._optical)
    {
        m_g4->preTrack();
    } 
}

void CTrackingAction::setTrack(const G4Track* track)
{
    m_ctx.setTrack(track);
}

void CTrackingAction::PostUserTrackingAction(const G4Track* track)
{
    int track_id = CTrack::Id(track) ;
    assert( track_id == m_ctx._track_id );
    assert( track == m_ctx._track );

    LOG(LEVEL) << brief()  ;

    if(m_ctx._optical)
    {
        m_g4->postTrack();
    } 
}

void CTrackingAction::dump(const char* msg )
{
    LOG(info) << msg  << " ctx " << m_ctx.desc() ; 
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

