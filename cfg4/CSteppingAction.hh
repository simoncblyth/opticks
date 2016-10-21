#pragma once


class G4ParticleDefinition ; 
class G4Track ; 
class G4Event ; 

#include "G4UserSteppingAction.hh"
#include "CBoundaryProcess.hh"
#include "globals.hh"

// CSteppingAction
// ================
//
//
//
// cg4-
class Opticks ; 
class CG4 ; 
class CPropLib ; 
class CRecorder ; 
class CGeometry ; 
class CMaterialBridge ; 
class CStepRec ; 


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSteppingAction : public G4UserSteppingAction
{
  static const unsigned long long SEQHIS_TO_SA ; 
  static const unsigned long long SEQMAT_MO_PY_BK ; 
  public:
    CSteppingAction(CG4* g4, bool dynamic);
    void postinitialize();
    void report(const char* msg="CSteppingAction::report");
    virtual ~CSteppingAction();

#ifdef USE_CUSTOM_BOUNDARY
    DsG4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
#else
    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
#endif
  public:
    virtual void UserSteppingAction(const G4Step*);
  private:
    void setEvent(const G4Event* event, int event_id);
    void setTrack(const G4Track* track, int track_id, int parent_id);
    bool setStep( const G4Step* step, int step_id);
    bool UserSteppingActionOptical(const G4Step* step);
    int getPrimaryPhotonID();

  private:
    CG4*         m_g4 ; 
    Opticks*     m_ok ; 

    bool         m_dynamic ; 
    CGeometry*   m_geometry ; 
    CMaterialBridge*  m_material_bridge ; 
    CPropLib*    m_clib ; 
    CRecorder*   m_recorder   ; 
    CStepRec*    m_steprec   ; 
    int          m_verbosity ; 

    unsigned int m_event_total ; 
    unsigned int m_track_total ; 
    unsigned int m_step_total ; 
    unsigned int m_event_track_count ; 
    unsigned int m_track_step_count ; 
    unsigned int m_steprec_store_count ;
    unsigned int m_rejoin_count ;

    const G4Event* m_event ; 
    const G4Track* m_track ; 
    const G4Step*  m_step ; 

    bool m_startEvent ; 
    bool m_startTrack ; 

    int m_event_id ;
    int m_track_id ;
    int m_parent_id ;
    int m_step_id ;
    int m_primary_id ; 

    G4TrackStatus         m_track_status ; 
    G4ParticleDefinition* m_particle  ; 
    bool                  m_optical ; 
    int                   m_pdg_encoding ;
 



};

#include "CFG4_TAIL.hh"

