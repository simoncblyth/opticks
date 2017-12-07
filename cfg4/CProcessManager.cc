
#include <sstream>
#include <iomanip>

#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "CProcessManager.hh"


std::string CProcessManager::Desc( G4ProcessManager* procMgr )
{
    std::stringstream ss ; 
    ss << "CProMgr"
       ;

    return ss.str();
}


G4ProcessManager* CProcessManager::Current(G4Track* trk) 
{
    const G4ParticleDefinition* defn = trk->GetDefinition() ;
    return defn->GetProcessManager() ;
}


