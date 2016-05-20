#pragma once

#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

class CPropLib ; 
class Recorder ; 
class Rec ; 

class CSteppingAction : public G4UserSteppingAction
{
  static const unsigned long long SEQHIS_TO_SA ; 
  static const unsigned long long SEQMAT_MO_PY_BK ; 

  public:
    CSteppingAction(CPropLib* clib, Recorder* recorder, Rec* rec, int verbosity=0);
    virtual ~CSteppingAction();

    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
    virtual void UserSteppingAction(const G4Step*);

  private:
    void init();

  private:
    CPropLib*    m_clib ; 
    Recorder*    m_recorder   ; 
    Rec*         m_rec   ; 
    int          m_verbosity ; 
};

inline CSteppingAction::CSteppingAction(CPropLib* clib, Recorder* recorder, Rec* rec, int verbosity)
   : 
   G4UserSteppingAction(),
   m_clib(clib),
   m_recorder(recorder),
   m_rec(rec),
   m_verbosity(verbosity)
{ 
   init();
}

inline CSteppingAction::~CSteppingAction()
{ 
}



