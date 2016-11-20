#pragma once

// CTrackingAction
// ================
//
//

#include <string>
#include "G4UserTrackingAction.hh"
#include "globals.hh"

class G4Event ; 

class Opticks ; 
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
  public:
    std::string brief(); 
  public:
    virtual void  PreUserTrackingAction(const G4Track* track);
    virtual void PostUserTrackingAction(const G4Track* track);
  private:
    void setEvent(const G4Event* event);
    void setTrack(const G4Track* track);
    void setPhotonId(int photon_id, bool reemtrack);
  private:
    CG4*               m_g4 ; 
    Opticks*           m_ok ; 
    CRecorder*         m_recorder ; 
    CSteppingAction*   m_sa ;  

  private:
    // setEvent
    const G4Event*        m_event ; 
    int                   m_event_id ;
  private:
    // setTrack
    const G4Track*        m_track ; 
    int                   m_track_id ;
    int                   m_parent_id ;
    G4TrackStatus         m_track_status ; 
    G4ParticleDefinition* m_particle  ; 
    int                   m_pdg_encoding ;
    bool                  m_optical ; 
    bool                  m_reemtrack ; 
    int                   m_primary_id ;
    int                   m_photon_id ;


};

#include "CFG4_TAIL.hh"

