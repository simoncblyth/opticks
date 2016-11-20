// g4-

#include "CFG4_PUSH.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"

// cg4-
#include "CG4.hh"
#include "CRecorder.hh"
#include "CTrack.hh"
#include "CTrackingAction.hh"

#include "PLOG.hh"


/**
CTrackingAction
=================

Canonical instance (m_sa) is ctor resident of CG4 

**/

CTrackingAction::CTrackingAction(CG4* g4)
   : 
   G4UserTrackingAction(),
   m_g4(g4),
   m_ok(g4->getOpticks()),
   m_recorder(g4->getRecorder()),

   m_track(NULL),
   m_track_id(-1),
   m_parent_id(-1),
   m_track_status(fAlive),
   m_particle(NULL),
   m_pdg_encoding(0),
   m_optical(false),
   m_optical_track_id(-1),
   m_optical_parent_id(-1)

{ 
}


CTrackingAction::~CTrackingAction()
{ 
}


void CTrackingAction::setTrack(const G4Track* track)
{
    m_track = track ; 
    m_track_id = CTrack::Id(track) ;
    m_parent_id = CTrack::ParentId(track) ;
    m_track_status = track->GetTrackStatus(); 

    m_particle = track->GetDefinition();
    m_optical = m_particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    m_pdg_encoding = m_particle->GetPDGEncoding();

    //m_event_track_count += 1 ; 
    //m_track_total += 1 ; 

    //m_track_step_count = 0 ; 
    //m_rejoin_count = 0 ; 

    if(m_optical)
    {
        m_optical_track_id = m_track_id ;
        m_optical_parent_id = m_parent_id ; 

        LOG(trace) << "CTrackingAction::setTrack(optical)"
                  << " optical_track_id " << m_optical_track_id
                  << " optical_parent_id " << m_optical_parent_id
                  ;

        if(m_optical_parent_id != -1 && m_optical_parent_id >= m_optical_track_id) 
        {
           LOG(fatal) << "CTrackingAction::setTrack(optical) UNEXPECTED m_optical_parent_id >= m_optical_track_id  "
                      << " optical_track_id " << m_optical_track_id
                      << " optical_parent_id " << m_optical_parent_id
                      ;

           assert(m_optical_parent_id < m_optical_track_id) ;  
        }
    }

}







void CTrackingAction::postinitialize()
{
    assert(m_track_id == -1);
    assert(m_parent_id == -1);
    LOG(trace) << "CTrackingAction::postinitialize" 
              << brief()
               ;
}


std::string CTrackingAction::brief()
{
    std::stringstream ss ; 
    ss  
       << " track_id " << m_track_id
       << " parent_id " << m_parent_id
       ;
    return ss.str();
}


void CTrackingAction::PreUserTrackingAction(const G4Track* track)
{
    setTrack(track);

    LOG(trace) << "CTrackingAction::PreUserTrackingAction"
              << brief()  
               ;
}

void CTrackingAction::PostUserTrackingAction(const G4Track* track)
{
    int track_id = CTrack::Id(track) ;
    assert( track_id == m_track_id );
    assert( track == m_track );

    LOG(trace) << "CTrackingAction::PostUserTrackingAction" 
              << brief() 
              ;

    if(m_optical)
    {
        m_recorder->posttrack();
    } 
}



