#pragma once

// CEventAction
// ================
//
//

#include <string>
#include "G4UserEventAction.hh"
#include "globals.hh"

class G4Event ; 

class Opticks ; 

class CG4 ; 
class CTrackingAction ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CEventAction : public G4UserEventAction
{
    friend class CG4 ;  
  public:
    virtual ~CEventAction();
  private:
    CEventAction(CG4* g4);
    void postinitialize();
  public:
    std::string brief(); 
  public:
    virtual void BeginOfEventAction(const G4Event* anEvent);
    virtual void EndOfEventAction(const G4Event* anEvent);
  private:
    void setEvent(const G4Event* event);
  private:
    CG4*               m_g4 ; 
    Opticks*           m_ok ; 
    CTrackingAction*   m_ta ;  
  private:
    // setEvent
    const G4Event*        m_event ; 
    int                   m_event_id ;

};

#include "CFG4_TAIL.hh"

