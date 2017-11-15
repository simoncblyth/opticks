// g4-
#include "CFG4_PUSH.hh"
#include "G4Event.hh"
#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"

// cg4-
#include "CG4Ctx.hh"
#include "CG4.hh"
#include "CEventAction.hh"

#include "PLOG.hh"

/**
CEventAction
=================

Canonical instance (m_ea) is ctor resident of CG4 

**/

CEventAction::CEventAction(CG4* g4)
   : 
   G4UserEventAction(),
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks())
{ 
}

CEventAction::~CEventAction()
{ 
}

void CEventAction::BeginOfEventAction(const G4Event* anEvent)
{
    setEvent(anEvent);
}

void CEventAction::EndOfEventAction(const G4Event* /*anEvent*/)
{
}

void CEventAction::setEvent(const G4Event* event)
{
    m_ctx.setEvent(event);
}

void CEventAction::postinitialize()
{
    LOG(trace) << "CEventAction::postinitialize" 
              << brief()
               ;
}

std::string CEventAction::brief()
{
    std::stringstream ss ; 
    ss  
       << " event_id " << m_ctx._event_id
       ;
    return ss.str();
}


