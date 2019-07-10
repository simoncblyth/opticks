#include "CRunAction.hh"
#include "Opticks.hh"
#include "PLOG.hh"

const plog::Severity CRunAction::LEVEL = PLOG::EnvLevel("CRunAction", "DEBUG"); 


CRunAction::CRunAction(OpticksHub* hub) 
   :
     G4UserRunAction(),
     m_hub(hub),
     m_count(0)
{
    LOG(info) << "count " << m_count   ;
}
CRunAction::~CRunAction()
{
    LOG(info) << "count " << m_count  ;
}
void CRunAction::BeginOfRunAction(const G4Run*)
{
    OKI_PROFILE("CRunAction::BeginOfRunAction");
    m_count += 1 ; 
    LOG(info) << "count " << m_count  ;
}
void CRunAction::EndOfRunAction(const G4Run*)
{
    OKI_PROFILE("CRunAction::EndOfRunAction"); 
    LOG(info) << "count " << m_count  ;
}



