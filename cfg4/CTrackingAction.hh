#pragma once

// CTrackingAction
// ================
//
//

#include <string>
#include "G4TrackStatus.hh"
#include "G4UserTrackingAction.hh"
#include "globals.hh"

class G4Event ; 

struct CG4Ctx ; 

class Opticks ; 
class OpticksEvent ; 
class CG4 ; 
class CRecorder ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CTrackingAction : public G4UserTrackingAction
{
    friend class CG4 ;  
  public:
    virtual ~CTrackingAction();
  private:
    CTrackingAction(CG4* g4);
    void postinitialize();
    void initEvent(OpticksEvent* evt);
  public:
    std::string brief(); 
  public:
    // invoked from CEventAction
    //void setEvent(const G4Event* event, int event_id);
    void setEvent();
  public:
    virtual void  PreUserTrackingAction(const G4Track* track);
    virtual void PostUserTrackingAction(const G4Track* track);
  private:
    void setTrack(const G4Track* track);
    void setPhotonId(int photon_id, bool reemtrack);
    //void setRecordId(int record_id);
    //void setDebug(bool dbg);
    //void setOther(bool other);
    void dump(const char* msg );

  private:
    CG4*                  m_g4 ; 
    CG4Ctx&               m_ctx ; 
    Opticks*              m_ok ; 
    CRecorder*            m_recorder ; 
    CSteppingAction*      m_sa ;  

  private:
    // setEvent
    //const G4Event*        m_event ; 
    //int                   m_event_id ;
  private:
    // setTrack
    //const G4Track*        m_track ; 
    //int                   m_track_id ;
    //int                   m_parent_id ;
    G4TrackStatus         m_track_status ; 
    //int                   m_pdg_encoding ;
    //bool                  m_optical ; 
    //bool                  m_reemtrack ; 
    bool                  m_dump ; 
    //int                   m_primary_id ;
    //int                   m_photon_id ;

    // initEvent
    //unsigned              m_photons_per_g4event ; 

    // setRecordId
    //int                   m_record_id ;
    //bool                  m_debug ; 
    //bool                  m_other ; 


};

#include "CFG4_TAIL.hh"

