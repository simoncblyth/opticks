// g4-

#include "CFG4_PUSH.hh"
//#include "G4RunManager.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"

#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"
#include "OpticksEvent.hh"

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

Canonical instance (m_ta) is ctor resident of CG4 

**/

CTrackingAction::CTrackingAction(CG4* g4)
   : 
   G4UserTrackingAction(),
   m_g4(g4),
   m_ok(g4->getOpticks()),
   m_recorder(g4->getRecorder()),
   m_sa(g4->getSteppingAction()),

   m_event(NULL),
   m_event_id(-1),

   m_track(NULL),
   m_track_id(-1),
   m_parent_id(-1),
   m_track_status(fAlive),
   m_particle(NULL),
   m_pdg_encoding(0),
   m_optical(false),
   m_reemtrack(false),
   m_dump(false),
   m_primary_id(-1),
   m_photon_id(-1),

   m_photons_per_g4event(0),

   m_record_id(-1),
   m_debug(false),
   m_other(false)
{ 
}


CTrackingAction::~CTrackingAction()
{ 
}

void CTrackingAction::initEvent(OpticksEvent* evt)
{
   // invoked from CG4::initEvent
    m_photons_per_g4event = evt->getNumPhotonsPerG4Event() ; 

    LOG(info) << "CTrackingAction::initEvent"
              << " photons_per_g4event " << m_photons_per_g4event
              ;
}


void CTrackingAction::setEvent(const G4Event* event, int event_id )
{
    m_event = event ; 
    m_event_id = event_id ;

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
         LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ; 
         const_cast<G4Track*>(m_track)->UseGivenVelocity(true);
         // NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
         //    are trumpled by G4Track::CalculateVelocity 

         int photon_id = -1 ; 
         int primary_id = CTrack::PrimaryPhotonID(m_track) ;    // layed down in trackinfo by custom Scintillation process
         bool reemtrack = false  ;
 
         if( primary_id >= 0)
         {
             reemtrack = true ; 
             photon_id = primary_id ;      // <-- tacking reem step recording onto primary record 
         }
         else
         {
             // with torch running aleays m_parent_id == -1 indicating primary photon
             // but remove that requirment for g4gun running

             // primary photon, ie not downstream from reemission 
             reemtrack = false ; 
             photon_id = m_track_id ; 
         }

         setPhotonId(photon_id, reemtrack);
         m_primary_id = primary_id ; // for debug only 
    }
}

void CTrackingAction::setPhotonId(int photon_id, bool reemtrack)
{
    m_photon_id = photon_id ;    // NB photon_id continues reemission photons
    m_reemtrack = reemtrack ; 

    m_sa->setPhotonId(m_photon_id, m_reemtrack);

    int record_id = m_photons_per_g4event*m_event_id + m_photon_id ; 
    setRecordId(record_id);

    if(m_dump) dump("CTrackingAction::setPhotonId");
}

void CTrackingAction::setRecordId(int record_id )
{
    m_record_id = record_id ; 

    bool _debug = m_ok->isDbgPhoton(record_id) ; // from option: --dindex=1,100,1000,10000 
    setDebug(_debug);

    bool other = m_ok->isOtherPhoton(record_id) ; // from option: --oindex=1,100,1000,10000 
    setOther(other);

    m_dump = m_debug || m_other ; 

    m_sa->setRecordId(record_id, _debug, other);
}

void CTrackingAction::setDebug(bool _debug)
{
    m_debug = _debug ; 
}
void CTrackingAction::setOther(bool other)
{
    m_other = other ; 
}

void CTrackingAction::dump(const char* msg )
{
    LOG(info) << "." ;
    LOG(info) << msg  
              << ( m_debug ? " --dindex " : "" )
              << ( m_other ? " --oindex " : "" )
              << " record_id " << m_record_id
              << " event_id " << m_event_id
              << " track_id " << m_track_id
              << " photon_id " << m_photon_id
              << " parent_id " << m_parent_id
              << " primary_id " << m_primary_id
              << " reemtrack " << m_reemtrack
              ;
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


