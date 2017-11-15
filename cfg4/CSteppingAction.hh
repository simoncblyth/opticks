#pragma once


class G4ParticleDefinition ; 
class G4Track ; 
class G4Event ; 

#include "G4ThreeVector.hh"
#include "G4UserSteppingAction.hh"
#include "CBoundaryProcess.hh"
#include "globals.hh"

/**

CSteppingAction
================


**/

// cg4-

class Opticks ; 

struct CG4Ctx ; 

class CG4 ; 
class CMaterialLib ; 
class CRecorder ; 
class CGeometry ; 
class CMaterialBridge ; 
class CStepRec ; 


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSteppingAction : public G4UserSteppingAction
{
   friend class CTrackingAction ; 

  public:
    CSteppingAction(CG4* g4, bool dynamic);
    void postinitialize();
    //int getStepId();
    void report(const char* msg="CSteppingAction::report");
    virtual ~CSteppingAction();

#ifdef USE_CUSTOM_BOUNDARY
    DsG4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
#else
    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
#endif
  public:
    //const G4ThreeVector& getStepOrigin();
  public:
    virtual void UserSteppingAction(const G4Step*);
  private:

    //void setEvent();
    //void setTrack();
    //void setEvent(const G4Event* event, int event_id);
    //void setTrack(const G4Track* track, int track_id, bool optical, int pdg_encoding );
    //void setPhotonId(int photon_id, bool reemtrack);
    //void setRecordId(int photon_id, bool debug, bool other);

    void setPhotonId();
    bool setStep( const G4Step* step, int step_id);
    bool collectPhotonStep();

  private:
    CG4*              m_g4 ; 
    CG4Ctx&           m_ctx ; 
    Opticks*          m_ok ; 
    bool              m_dbgrec ; 
    bool              m_dynamic ; 
    CGeometry*        m_geometry ; 
    CMaterialBridge*  m_material_bridge ; 
    CMaterialLib*     m_mlib ; 
    CRecorder*        m_recorder   ; 
    CStepRec*         m_steprec   ; 
    int               m_verbosity ; 

    // init in ctor
    //unsigned int m_event_total ; 
    //unsigned int m_track_total ; 
    //unsigned int m_step_total ; 
    //unsigned int m_event_track_count ; 
    unsigned int m_steprec_store_count ;

    // set by UserSteppingAction
    //bool                  m_startEvent ; 
    //bool                  m_startTrack ; 

    // set by setEvent
    //const G4Event*        m_event ; 
    //int                   m_event_id ;

    // set by setTrack
    //unsigned int          m_track_step_count ; 

    //const G4Track*        m_track ; 
    //int                   m_track_id ;
    //bool                  m_optical ; 
    //int                   m_pdg_encoding ;

    // set by setPhotonId
    //int                   m_photon_id ; 
    //bool                  m_reemtrack ; 

    //unsigned int          m_rejoin_count ;
    //unsigned int          m_primarystep_count ;
    
    // set by setRecordId
    //int                    m_record_id ;
    //bool                   m_debug ; 
    //bool                   m_other ; 

  
    // set by setStep 
    //const G4Step*         m_step ; 
    //int                   m_step_id ;
    //G4ThreeVector         m_step_origin ; 



};

#include "CFG4_TAIL.hh"

