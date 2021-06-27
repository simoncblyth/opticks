#include "CScint.hh"
#include "G4OpticalPhoton.hh"
#include "G4ProcessManager.hh"

#include "CProcessManager.hh"

#include "PLOG.hh"


void CScint::Check()
{
    G4OpticalPhoton* particle =  G4OpticalPhoton::Definition() ; 
    G4ProcessManager* pmanager = particle->GetProcessManager(); 
    G4VProcess* proc = pmanager->GetProcess("Scintillation"); 

    LOG(info) 
        << " pmanager " << pmanager 
        << " proc " << proc 
        ;

    LOG(info) << CProcessManager::Desc(pmanager) ; 


}
