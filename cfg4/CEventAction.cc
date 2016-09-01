#include "CEventAction.hh"
#include "PLOG.hh"

CEventAction::CEventAction(OpticksHub* hub) 
    :
    G4UserEventAction(),
    m_hub(hub),
    m_count(0)
{
    LOG(info) << "CEventAction::CEventAction count " << m_count  ;
}
CEventAction::~CEventAction()
{
    LOG(info) << "CEventAction::~CEventAction count " << m_count  ;
}
void CEventAction::BeginOfEventAction(const G4Event*)
{
   m_count += 1 ; 
   LOG(info) << "CEventAction::BeginOfEventAction count " << m_count ;
}
void CEventAction::EndOfEventAction(const G4Event*)
{
   LOG(info) << "CEventAction::EndOfEventAction count " << m_count  ;
}



