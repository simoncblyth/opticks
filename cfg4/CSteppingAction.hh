#pragma once


#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

// CSteppingAction
// ================
//
//
//
// cg4-
class CG4 ; 
class CPropLib ; 
class CRecorder ; 
class Rec ; 
class CStepRec ; 


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSteppingAction : public G4UserSteppingAction
{
  static const unsigned long long SEQHIS_TO_SA ; 
  static const unsigned long long SEQMAT_MO_PY_BK ; 

  public:
    CSteppingAction(CG4* g4, bool dynamic);
    virtual ~CSteppingAction();

    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
    virtual void UserSteppingAction(const G4Step*);
  private:
    void init();
    void UserSteppingActionOptical(const G4Step*);
    void compareRecords();
    void setTrackId(unsigned int track_id);
    void setEventId(unsigned int event_id);
  private:
    CG4*         m_g4 ; 
    bool         m_dynamic ; 
    CPropLib*    m_clib ; 
    CRecorder*   m_recorder   ; 
    Rec*         m_rec   ; 
    CStepRec*    m_steprec   ; 
    int          m_verbosity ; 
    int          m_event_id ; 
    int          m_track_id ; 
    unsigned int m_event_total ; 
    unsigned int m_track_total ; 
    unsigned int m_step_total ; 
    unsigned int m_event_track_count ; 
    unsigned int m_track_step_count ; 
    unsigned int m_steprec_store_count ;
};

#include "CFG4_TAIL.hh"

