// g4-

#include "CFG4_PUSH.hh"
#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"

#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"

// cg4-
#include "CG4.hh"
#include "CRecorder.hh"
#include "CTrack.hh"
#include "CSteppingAction.hh"
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
   m_sa(g4->getSteppingAction()),

   m_track(NULL),
   m_track_id(-1),
   m_parent_id(-1),
   m_track_status(fAlive),
   m_particle(NULL),
   m_pdg_encoding(0),
   m_optical(false),
   m_reemtrack(false),
   m_primary_id(-1),
   m_photon_id(-1)
{ 
}


CTrackingAction::~CTrackingAction()
{ 
}

void CTrackingAction::setEvent(const G4Event* event)
{
    m_event = event ; 
    m_event_id = event->GetEventID() ;

    m_sa->setEvent(m_event, m_event_id );
}

void CTrackingAction::setTrack(const G4Track* track)
{
    m_track = track ; 
    m_track_id = CTrack::Id(track) ;
    m_parent_id = CTrack::ParentId(track) ;

    if(m_parent_id != -1 && m_parent_id >= m_track_id) 
    {
       LOG(fatal) << "CTrackingAction::setTrack UNEXPECTED m_parent_id >= m_track_id  "
                  << " track_id " << m_track_id
                  << " parent_id " << m_parent_id
                  ;

       assert(m_parent_id < m_track_id) ;  
    }

    m_track_status = track->GetTrackStatus(); 

    m_particle = track->GetDefinition();
    m_optical = m_particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    m_pdg_encoding = m_particle->GetPDGEncoding();


    m_sa->setTrack(m_track, m_track_id, m_optical, m_pdg_encoding );


    if(m_optical)
    { 
         int photon_id = -1 ; 
         int primary_id = CTrack::PrimaryPhotonID(m_track) ;    // layed down in trackinfo by custom Scintillation process
         bool reemtrack = false  ;
 
         if( m_parent_id == -1 )     // primary photon, ie not downstream from reemission 
         {
             reemtrack = false ; 
             photon_id = m_track_id ; 
         } 
         else if( primary_id >= 0)
         {
             reemtrack = true ; 
             photon_id = primary_id ;      // <-- tacking reem step recording onto primary record 
         }
         else
         {
             assert(0);
         } 

         setPhotonId(photon_id, reemtrack);
         m_primary_id = primary_id ; // for debug only 
    }
}

void CTrackingAction::setPhotonId(int photon_id, bool reemtrack)
{
    m_photon_id = photon_id ; 
    m_reemtrack = reemtrack ; 

    LOG(info) << "." ;
    LOG(info) << "." ;
    LOG(info) << "CTrackingAction::setPhotonId"
              << " track_id " << m_track_id
              << " parent_id " << m_parent_id
              << " primary_id " << m_primary_id
              << " photon_id " << m_photon_id
              << " reemtrack " << m_reemtrack
              ;

    m_sa->setPhotonId(m_photon_id, m_reemtrack);
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

    const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;

    setEvent(event);

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



