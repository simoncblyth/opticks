#include "CRunAction.hh"
#include "PLOG.hh"

CRunAction::CRunAction(OpticksHub* hub) 
   :
     G4UserRunAction(),
     m_hub(hub),
     m_count(0)
{
    LOG(info) << "CRunAction::CRunAction count " << m_count   ;
}
CRunAction::~CRunAction()
{
    LOG(info) << "CRunAction::~CRunAction count " << m_count  ;
}
void CRunAction::BeginOfRunAction(const G4Run*)
{
    m_count += 1 ; 
    LOG(info) << "CRunAction::BeginOfRunAction count " << m_count  ;
}
void CRunAction::EndOfRunAction(const G4Run*)
{
    LOG(info) << "CRunAction::EndOfRunAction count " << m_count  ;
}



