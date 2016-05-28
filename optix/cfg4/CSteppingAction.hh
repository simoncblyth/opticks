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
class Recorder ; 
class Rec ; 
class CStepRec ; 

class CSteppingAction : public G4UserSteppingAction
{
  static const unsigned long long SEQHIS_TO_SA ; 
  static const unsigned long long SEQMAT_MO_PY_BK ; 

  public:
    CSteppingAction(CG4* g4);
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
    CPropLib*    m_clib ; 
    Recorder*    m_recorder   ; 
    Rec*         m_rec   ; 
    CStepRec*    m_steprec   ; 
    bool         m_dynamic ; 
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

inline CSteppingAction::CSteppingAction(CG4* g4)
   : 
   G4UserSteppingAction(),
   m_g4(g4),
   m_clib(NULL),
   m_recorder(NULL),
   m_rec(NULL),
   m_steprec(NULL),
   m_dynamic(false),
   m_verbosity(0),
   m_event_id(-1),
   m_track_id(-1),
   m_event_total(0),
   m_track_total(0),
   m_step_total(0),
   m_event_track_count(0),
   m_track_step_count(0),
   m_steprec_store_count(0)
{ 
   init();
}

inline CSteppingAction::~CSteppingAction()
{ 
}


inline void CSteppingAction::setTrackId(unsigned int track_id)
{
    m_track_id = track_id ; 
}
inline void CSteppingAction::setEventId(unsigned int event_id)
{
    m_event_id = event_id ; 
}

